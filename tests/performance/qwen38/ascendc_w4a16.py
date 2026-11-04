"""独立AscendC W4A16原型。调用者持有工作区；不改生产注册表。"""
import ctypes
from pathlib import Path

import torch
import torch_npu


def pack_k_int4(integer):
    """将[N,K]有符号INT4按K轴相邻两项打包，区别于模型文件的N轴打包。"""
    if integer.ndim != 2 or integer.shape[1] % 128 or integer.dtype != torch.int8:
        raise ValueError('权重要求int8载体和128对齐的K维')
    if bool((integer < -8).any()) or bool((integer > 7).any()):
        raise ValueError('权重超出INT4范围')
    return ((integer[:, 0::2] & 15) | ((integer[:, 1::2] & 15) << 4)).to(torch.uint8).contiguous()


class AscendCW4A16:
    def __init__(self, library, tiling=None, *, pipeline=False):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.tiling = tiling
        self.pipeline = pipeline
        if tiling is not None and (len(tiling) != 3 or tiling[0] not in (64,128)
                                  or tiling[1] not in (256,512) or tiling[2] not in (1,2,4,8)):
            raise ValueError('未编译的分块或Split-K配置')
        if pipeline and (tiling is None or tiling[2] != 1):
            raise ValueError('流水候选要求显式分块且Split-K=1')
        self.run = (self.library.w4a16_pipeline if pipeline else
                    self.library.w4a16_run if tiling is None else self.library.w4a16_tuned)
        self.run.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_uint32] * 3 if tiling is None
                             else [ctypes.c_void_p] * 7 + [ctypes.c_uint32] * 6)
        if pipeline:
            self.run.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_uint32]*5
        self.run.restype = None

    def __call__(self, x, packed, scales, workspace, output, partial=None):
        m, k = x.shape
        n = output.shape[1]
        if (x.dtype != torch.bfloat16 or scales.dtype != torch.bfloat16
                or workspace.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
                or packed.dtype != torch.uint8 or k % 128 or packed.shape != (n, k // 2)
                or scales.shape != (n, k // 128) or workspace.shape != (n, k)
                or output.shape != (m, n) or not all(t.is_contiguous() for t in (x, packed, scales, workspace, output))
                or not all(t.device == x.device for t in (packed, scales, workspace, output))
                or x.device.type != 'npu'):
            raise ValueError('W4A16要求同一NPU的连续BF16张量、行打包INT4和group128尺度')
        if any(torch_npu.get_npu_format(t) != 2 for t in (x,packed,scales,workspace,output)):
            raise ValueError('原型原始指针接口仅接受ND存储，不能将NZ张量当作ND读取')
        stream = torch.npu.current_stream(x.device).npu_stream
        if self.pipeline:
            if workspace.numel() < 40*self.tiling[0]*k or workspace.numel()*2 > 1024**3:
                raise ValueError('流水候选要求20组双缓冲工作区且总量不超过1GiB')
            self.run(stream,x.data_ptr(),packed.data_ptr(),scales.data_ptr(),workspace.data_ptr(),
                     output.data_ptr(),m,n,k,*self.tiling[:2])
        elif self.tiling is None:
            self.run(stream, x.data_ptr(), packed.data_ptr(), scales.data_ptr(),
                     workspace.data_ptr(), output.data_ptr(), m, n, k)
        else:
            split = self.tiling[2]
            if (k % (split*128) or partial is None or partial.dtype != torch.float32
                    or partial.device != x.device or not partial.is_contiguous()
                    or torch_npu.get_npu_format(partial) != 2
                    or partial.numel() < (split*m*n if split > 1 else 1)
                    or workspace.numel()*2 + partial.numel()*4 > 1024**3):
                raise ValueError('Split-K工作区不满足对齐、容量或1GiB上限')
            self.run(stream,x.data_ptr(),packed.data_ptr(),scales.data_ptr(),workspace.data_ptr(),
                     partial.data_ptr(),output.data_ptr(),m,n,k,*self.tiling)
        return output


class AscendCNZW4A16:
    """直接消费现有NZ INT4权重及[K/group,N]尺度，不创建持久权重副本。"""
    def __init__(self, library, tiling):
        if tuple(tiling) not in [(n,k,1) for n in (64,128) for k in (256,512)]:
            raise ValueError('NZ流水仅支持已编译的分块及Split-K=1')
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.tiling = tuple(tiling)
        self.run = self.library.w4a16_nz_pipeline
        self.run.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_uint32]*5
        self.run.restype = None

    def __call__(self, x, packed, scales, workspace, output, partial=None):
        m,k=x.shape
        n=output.shape[1]
        tensors=(x,packed,scales,workspace,output)
        if (x.device.type!='npu' or any(t.device!=x.device for t in tensors)
                or any(not t.is_contiguous() for t in tensors)
                or packed.dtype!=torch.int32 or packed.shape!=(k,n//8)
                or k%128 or n%128 or scales.shape!=(k//128,n)
                or output.shape!=(m,n)
                or any(t.dtype!=torch.bfloat16 for t in (x,scales,workspace,output))
                or torch_npu.get_npu_format(packed)!=29
                or any(torch_npu.get_npu_format(t)!=2 for t in (x,scales,workspace,output))
                or workspace.numel()<40*self.tiling[0]*k or workspace.numel()*2>1024**3):
            raise ValueError('NZ流水要求原有NZ/int32载体、group128尺度和有界ND双缓冲')
        self.run(torch.npu.current_stream(x.device).npu_stream,
                 x.data_ptr(),packed.data_ptr(),scales.data_ptr(),workspace.data_ptr(),output.data_ptr(),
                 m,n,k,*self.tiling[:2])
        return output
