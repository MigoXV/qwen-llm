"""V2布局及有限调优空间；纯CPU部分不初始化设备。"""
from dataclasses import dataclass, asdict
from enum import IntEnum
import itertools


class Layout(IntEnum):
    NATIVE_NZ = 0
    BLOCKED = 1


LAYOUT_VERSION = 'blocked64x128-v1'
SOURCE_FILES = ('verify_ascendc_v2.py','ascendc_v2.py','ascendc_v2_runtime.py',
                'ascendc_v2_candidate.py','quantization.py')


def pack_blocked(integer):
    import torch
    if (integer.dtype != torch.int8 or integer.ndim != 2
            or integer.shape[0] % 64 or integer.shape[1] % 128
            or bool(((integer < -8) | (integer > 7)).any())):
        raise ValueError('布局要求[N,K]有符号INT4、N64/K128对齐')
    n,k=integer.shape
    # [Ntile,Kgroup,K16,N16,Ninner,Kinner], matching BF16 nZ(128,64).
    flat=integer.t().reshape(k//128,8,16,n//64,4,16).permute(3,0,1,4,5,2).contiguous().flatten()
    return ((flat[::2]&15) | ((flat[1::2]&15)<<4)).to(torch.uint8)


def unpack_blocked(packed,n,k):
    import torch
    if packed.dtype != torch.uint8 or packed.numel()*2 != n*k or n%64 or k%128:
        raise ValueError('专用布局容量或形状错误')
    flat=torch.stack((packed&15,packed>>4),dim=-1).flatten().to(torch.int8)
    flat=torch.where(flat>=8,flat-16,flat)
    return flat.reshape(n//64,k//128,8,4,16,16).permute(1,2,5,0,3,4).reshape(k,n).t().contiguous()


def index_maps():
    """uint16 Gather的字节偏移：nZ→ND，广播尺度，逆打包四列的基址。"""
    import torch
    i=torch.arange(8192,dtype=torch.int32)
    nd=(((i&15)+(i//1024)*16)*64+(i//16)%64)*2
    j=torch.arange(2048,dtype=torch.int32)
    scale=((j//16)%64)*2
    t=torch.arange(512,dtype=torch.int32)
    row=t//16; col=t%16
    inverse=((row//4)*256+col*16+row%4)*2
    return torch.cat((nd,scale,inverse)).contiguous()


@dataclass(frozen=True)
class Tiling:
    ntile:int=128
    ktile:int=256
    mtile:int=16
    depth:int=2
    split:int=1
    family:int=1  # 0: batched dequant + GEMM; 1: rolling-K mixed GEMM.

    def __post_init__(self):
        if (self.ntile not in (64,128) or self.ktile not in (128,256,512)
                or self.mtile not in (16,64,128,256) or self.depth not in (2,3)
                or self.split not in (1,2,4,8) or self.family not in (0,1)):
            raise ValueError('未编译的V2分块')
        if ((self.mtile+self.ntile)*self.ktile*4>512*1024
                or self.mtile*64*4>64*1024 or self.mtile*self.ntile*4>128*1024):
            raise ValueError('分块超过L1/L0预算')
        if self.family==0 and self.split!=1:
            raise ValueError('批量反量化对照不使用Split-K')

    def workspace_bytes(self,m,n,k):
        b=(n*k*2 if self.family==0 else 20*self.depth*self.ntile*self.ktile*2)
        partial=self.split*m*n*4 if self.split>1 else 0
        return b+partial

    def values(self):
        return tuple(asdict(self).values())


def configurations(m):
    sizes=(16,) if m<=16 else (64,) if m<=64 else (128,) if m<1536 else (64,128,256)
    for mt,nt,kt,depth in itertools.product(sizes,(64,128),(128,256,512),(2,3)):
        try:yield Tiling(nt,kt,mt,depth)
        except ValueError:continue
    for mt,nt,kt in itertools.product(sizes,(64,128),(128,256,512)):
        try:yield Tiling(nt,kt,mt,2,1,0)
        except ValueError:continue
