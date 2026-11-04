// Experimental group128 W4A16: explicit dequantization workspace + CATLASS GEMM.
// This is a comparison prototype, not a claim of fused/pipelined execution.
#include "kernel_operator.h"
#include "adv_api/quantization/ascend_antiquant.h"
#include "kernel/basic_matmul.hpp"
#include <acl/acl.h>
using namespace AscendC;

extern "C" __global__ __aicore__ void unpack_group128(GM_ADDR packed, GM_ADDR scales,
                                                     GM_ADDR output, uint32_t groups) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> srcBuf, dstBuf, tmpBuf;
    pipe.InitBuffer(srcBuf, 64);
    pipe.InitBuffer(dstBuf, 256);
    pipe.InitBuffer(tmpBuf, 4096);
    auto src = srcBuf.Get<int4b_t>();
    auto dst = dstBuf.Get<bfloat16_t>();
    auto tmp = tmpBuf.Get<uint8_t>();
    GlobalTensor<uint8_t> w;
    GlobalTensor<bfloat16_t> s, y;
    w.SetGlobalBuffer((__gm__ uint8_t*)packed);
    s.SetGlobalBuffer((__gm__ bfloat16_t*)scales);
    y.SetGlobalBuffer((__gm__ bfloat16_t*)output);
    for (uint32_t g = GetBlockIdx(); g < groups; g += GetBlockNum()) {
        DataCopy(src.ReinterpretCast<uint8_t>(), w[g * 64], 64);
        auto event = pipe.FetchEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);
        AscendAntiQuant<int4b_t, bfloat16_t, false>(dst, src, s.GetValue(g), tmp, 128);
        auto write = pipe.FetchEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(write);
        WaitFlag<HardEvent::V_MTE3>(write);
        DataCopy(y[g * 128], dst, 128);
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" void w4a16_run(void* stream, void* x, void* w, void* s, void* workspace,
                           void* y, uint32_t m, uint32_t n, uint32_t k) {
    unpack_group128<<<40, nullptr, stream>>>((GM_ADDR)w, (GM_ADDR)s, (GM_ADDR)workspace, n * k / 128);
    Catlass::basic_matmul<Catlass::layout::RowMajor, Catlass::layout::ColumnMajor,
        Catlass::layout::RowMajor, bfloat16_t, bfloat16_t><<<20, nullptr, stream>>>(
        Catlass::GemmCoord{m, n, k}, (GM_ADDR)x, (GM_ADDR)workspace, (GM_ADDR)y);
}

// Shape-specific GEMM variants. A split changes only the reduction partition;
// leading dimensions still refer to the original tensors.
template<int NTile, int KTile, typename Out>
__global__ __aicore__ void split_gemm(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                                    uint32_t m, uint32_t n, uint32_t k,
                                    uint32_t split, uint32_t part) {
    using namespace Catlass;
    using Policy = Gemm::MmadAtlasA2Pingpong<true>;
    using Shape = GemmShape<128, NTile, KTile>;
    using L0 = GemmShape<128, NTile, 64>;
    using A = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
    using B = Gemm::GemmType<bfloat16_t, layout::ColumnMajor>;
    using C = Gemm::GemmType<Out, layout::RowMajor>;
    using Block = Gemm::Block::BlockMmad<Policy, Shape, L0, A, B, C>;
    using Swizzle = Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
    using Kernel = Gemm::Kernel::BasicMatmul<Block, void, Swizzle>;
    uint32_t chunk = k / split;
    GemmCoord shape{m, n, chunk};
    typename Kernel::Params params{shape, x + part * chunk * 2,
        layout::RowMajor(m, k), w + part * chunk * 2,
        layout::ColumnMajor(k, n), y, layout::RowMajor(m, n)};
    Kernel kernel;
    kernel(params);
}

extern "C" __global__ __aicore__ void reduce_parts(GM_ADDR input, GM_ADDR output,
                                                  uint32_t count, uint32_t split) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> sumBuf, tmpBuf, outBuf;
    pipe.InitBuffer(sumBuf, 512);
    pipe.InitBuffer(tmpBuf, 512);
    pipe.InitBuffer(outBuf, 256);
    auto sum = sumBuf.Get<float>();
    auto tmp = tmpBuf.Get<float>();
    auto out = outBuf.Get<bfloat16_t>();
    GlobalTensor<float> src;
    GlobalTensor<bfloat16_t> dst;
    src.SetGlobalBuffer((__gm__ float*)input);
    dst.SetGlobalBuffer((__gm__ bfloat16_t*)output);
    for (uint32_t offset = GetBlockIdx() * 128; offset < count; offset += GetBlockNum() * 128) {
        Duplicate(sum, 0.0f, 128);
        for (uint32_t p = 0; p < split; ++p) {
            DataCopy(tmp, src[p * count + offset], 128);
            auto event = pipe.FetchEventID<HardEvent::MTE2_V>();
            SetFlag<HardEvent::MTE2_V>(event);
            WaitFlag<HardEvent::MTE2_V>(event);
            Add(sum, sum, tmp, 128);
            PipeBarrier<PIPE_ALL>();
        }
        Cast(out, sum, RoundMode::CAST_RINT, 128);
        auto event = pipe.FetchEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
        DataCopy(dst[offset], out, 128);
        PipeBarrier<PIPE_ALL>();
    }
}

template<int NTile, int KTile>
void launch_parts(void* stream, GM_ADDR x, GM_ADDR workspace, GM_ADDR partial,
                  GM_ADDR y, uint32_t m, uint32_t n, uint32_t k, uint32_t split) {
    if (split == 1) {
        split_gemm<NTile, KTile, bfloat16_t><<<20, nullptr, stream>>>(x, workspace, y, m, n, k, 1, 0);
    } else {
        for (uint32_t part = 0; part < split; ++part) {
            split_gemm<NTile, KTile, float><<<20, nullptr, stream>>>(x, workspace,
                partial + part * m * n * 4, m, n, k, split, part);
        }
        reduce_parts<<<40, nullptr, stream>>>(partial, y, m * n, split);
    }
}

extern "C" void w4a16_tuned(void* stream, void* x, void* w, void* s, void* workspace,
    void* partial, void* y, uint32_t m, uint32_t n, uint32_t k,
    uint32_t ntile, uint32_t ktile, uint32_t split) {
    unpack_group128<<<40, nullptr, stream>>>((GM_ADDR)w, (GM_ADDR)s, (GM_ADDR)workspace, n*k/128);
    if (ntile == 64 && ktile == 256)
        launch_parts<64,256>(stream,(GM_ADDR)x,(GM_ADDR)workspace,(GM_ADDR)partial,(GM_ADDR)y,m,n,k,split);
    else if (ntile == 64 && ktile == 512)
        launch_parts<64,512>(stream,(GM_ADDR)x,(GM_ADDR)workspace,(GM_ADDR)partial,(GM_ADDR)y,m,n,k,split);
    else if (ntile == 128 && ktile == 256)
        launch_parts<128,256>(stream,(GM_ADDR)x,(GM_ADDR)workspace,(GM_ADDR)partial,(GM_ADDR)y,m,n,k,split);
    else if (ntile == 128 && ktile == 512)
        launch_parts<128,512>(stream,(GM_ADDR)x,(GM_ADDR)workspace,(GM_ADDR)partial,(GM_ADDR)y,m,n,k,split);
}

#include "catlass/arch/cross_core_sync.hpp"

// One AIC and its two AIV partners share two bounded GM B tiles. The AIVs
// prepare tile i+1 while the AIC consumes tile i. Every ready/free flag is
// consumed, including the final two acknowledgements, before kernel exit.
template<int NTile, int KTile>
__global__ __aicore__ void pipeline_gemm(GM_ADDR x, GM_ADDR packed, GM_ADDR scales,
                                       GM_ADDR staging, GM_ADDR output,
                                       uint32_t m, uint32_t n, uint32_t k) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using namespace Catlass;
    Arch::CrossCoreFlag ready0(0), ready1(1), free0(2), free1(3);
    if ASCEND_IS_AIV {
        uint32_t core = GetBlockIdx() / GetSubBlockNum();
        uint32_t sub = GetSubBlockIdx();
        TPipe pipe;
        TBuf<TPosition::VECCALC> srcBuf, dstBuf, tmpBuf;
        pipe.InitBuffer(srcBuf,64);
        pipe.InitBuffer(dstBuf,256);
        pipe.InitBuffer(tmpBuf,4096);
        auto src = srcBuf.Get<int4b_t>();
        auto dst = dstBuf.Get<bfloat16_t>();
        auto tmp = tmpBuf.Get<uint8_t>();
        GlobalTensor<uint8_t> w;
        GlobalTensor<bfloat16_t> s, b;
        w.SetGlobalBuffer((__gm__ uint8_t*)packed);
        s.SetGlobalBuffer((__gm__ bfloat16_t*)scales);
        b.SetGlobalBuffer((__gm__ bfloat16_t*)staging);
        uint32_t iteration = 0;
        for (uint32_t col = core*NTile; col < n; col += 20*NTile, ++iteration) {
            uint32_t slot = iteration % 2;
            if (iteration >= 2) {
                if (slot == 0) Arch::CrossCoreWaitFlag(free0);
                else Arch::CrossCoreWaitFlag(free1);
            }
            for (uint32_t local = sub*(NTile/2); local < (sub+1)*(NTile/2); ++local) {
                uint32_t row = col + local;
                if (row >= n) continue;
                for (uint32_t group = 0; group < k/128; ++group) {
                    DataCopy(src.ReinterpretCast<uint8_t>(),w[row*k/2+group*64],64);
                    auto event = pipe.FetchEventID<HardEvent::MTE2_V>();
                    SetFlag<HardEvent::MTE2_V>(event);
                    WaitFlag<HardEvent::MTE2_V>(event);
                    AscendAntiQuant<int4b_t,bfloat16_t,false>(dst,src,s.GetValue(row*(k/128)+group),tmp,128);
                    auto write = pipe.FetchEventID<HardEvent::V_MTE3>();
                    SetFlag<HardEvent::V_MTE3>(write);
                    WaitFlag<HardEvent::V_MTE3>(write);
                    DataCopy(b[((core*2+slot)*NTile+local)*k+group*128],dst,128);
                    PipeBarrier<PIPE_ALL>();
                }
            }
            if (slot == 0) Arch::CrossCoreSetFlag<0x2,PIPE_MTE3>(ready0);
            else Arch::CrossCoreSetFlag<0x2,PIPE_MTE3>(ready1);
        }
        uint32_t first = iteration > 2 ? iteration-2 : 0;
        for (uint32_t i=first; i<iteration; ++i) {
            if (i%2 == 0) Arch::CrossCoreWaitFlag(free0);
            else Arch::CrossCoreWaitFlag(free1);
        }
    }
    if ASCEND_IS_AIC {
        uint32_t core = GetBlockIdx();
        using Policy = Gemm::MmadAtlasA2Pingpong<true>;
        using Shape = GemmShape<128,NTile,KTile>;
        using L0 = GemmShape<128,NTile,64>;
        using A = Gemm::GemmType<bfloat16_t,layout::RowMajor>;
        using B = Gemm::GemmType<bfloat16_t,layout::ColumnMajor>;
        using C = Gemm::GemmType<bfloat16_t,layout::RowMajor>;
        using Block = Gemm::Block::BlockMmad<Policy,Shape,L0,A,B,C>;
        Arch::Resource<Arch::AtlasA2> resource;
        Block matmul(resource);
        GlobalTensor<bfloat16_t> a,b,c;
        a.SetGlobalBuffer((__gm__ bfloat16_t*)x);
        b.SetGlobalBuffer((__gm__ bfloat16_t*)staging);
        c.SetGlobalBuffer((__gm__ bfloat16_t*)output);
        uint32_t iteration=0;
        for (uint32_t col=core*NTile; col<n; col+=20*NTile,++iteration) {
            uint32_t slot=iteration%2;
            if (slot==0) Arch::CrossCoreWaitFlag(ready0);
            else Arch::CrossCoreWaitFlag(ready1);
            uint32_t cols = n-col < NTile ? n-col : NTile;
            for (uint32_t row=0;row<m;row+=128) {
                uint32_t rows = m-row < 128 ? m-row : 128;
                matmul(a[row*k],layout::RowMajor(m,k),b[(core*2+slot)*NTile*k],
                    layout::ColumnMajor(k,NTile),c[row*n+col],layout::RowMajor(m,n),
                    GemmCoord{rows,cols,k});
            }
            PipeBarrier<PIPE_ALL>();
            if (slot==0) Arch::CrossCoreSetFlag<0x2,PIPE_FIX>(free0);
            else Arch::CrossCoreSetFlag<0x2,PIPE_FIX>(free1);
        }
    }
}

extern "C" void w4a16_pipeline(void* stream, void* x, void* w, void* s, void* workspace,
    void* y, uint32_t m, uint32_t n, uint32_t k, uint32_t ntile, uint32_t ktile) {
    if (ntile==64 && ktile==256)
        pipeline_gemm<64,256><<<20,nullptr,stream>>>((GM_ADDR)x,(GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)workspace,(GM_ADDR)y,m,n,k);
    else if (ntile==64 && ktile==512)
        pipeline_gemm<64,512><<<20,nullptr,stream>>>((GM_ADDR)x,(GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)workspace,(GM_ADDR)y,m,n,k);
    else if (ntile==128 && ktile==256)
        pipeline_gemm<128,256><<<20,nullptr,stream>>>((GM_ADDR)x,(GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)workspace,(GM_ADDR)y,m,n,k);
    else if (ntile==128 && ktile==512)
        pipeline_gemm<128,512><<<20,nullptr,stream>>>((GM_ADDR)x,(GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)workspace,(GM_ADDR)y,m,n,k);
}

// NZ input retains the original vLLM packed weight and merged BF16 scales.
// INT8 NZ tiles hold 16 K rows x 32 packed N bytes = 16 x 64 INT4 values.
// Dequantized staging is [K,NTile] ND, so no persistent weight copy is needed.
template<int MTile, int NTile, int KTile>
__global__ __aicore__ void pipeline_nz_gemm(GM_ADDR x, GM_ADDR packed, GM_ADDR scales,
    GM_ADDR staging, GM_ADDR output, uint32_t m, uint32_t n, uint32_t k) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using namespace Catlass;
    Arch::CrossCoreFlag ready0(0),ready1(1),free0(2),free1(3);
    if ASCEND_IS_AIV {
        uint32_t core=GetBlockIdx()/GetSubBlockNum(), sub=GetSubBlockIdx();
        TPipe pipe;
        TBuf<TPosition::VECCALC> srcBuf,dstBuf,scaleBuf,tmpBuf;
        pipe.InitBuffer(srcBuf,512); pipe.InitBuffer(dstBuf,2048);
        pipe.InitBuffer(scaleBuf,128); pipe.InitBuffer(tmpBuf,16384);
        auto src=srcBuf.Get<int4b_t>(); auto dst=dstBuf.Get<bfloat16_t>();
        auto scale=scaleBuf.Get<bfloat16_t>(); auto tmp=tmpBuf.Get<uint8_t>();
        GlobalTensor<uint8_t> w;
        GlobalTensor<bfloat16_t> s,b;
        w.SetGlobalBuffer((__gm__ uint8_t*)packed);
        s.SetGlobalBuffer((__gm__ bfloat16_t*)scales);
        b.SetGlobalBuffer((__gm__ bfloat16_t*)staging);
        AntiQuantShapeInfo info{0,0,1,64};
        uint32_t iteration=0;
        for (uint32_t col=core*NTile;col<n;col+=20*NTile,++iteration) {
            uint32_t slot=iteration%2;
            if (iteration>=2) {
                if(slot==0) Arch::CrossCoreWaitFlag(free0); else Arch::CrossCoreWaitFlag(free1);
            }
            uint32_t localCol = NTile==128 ? sub*64 : 0;
            uint32_t beginK = NTile==64 ? sub : 0;
            uint32_t stepK = NTile==64 ? 2 : 1;
            for (uint32_t kb=beginK;kb<k/16;kb+=stepK) {
                uint32_t nz=((col+localCol)/64)*(k/16)+kb;
                DataCopy(src.ReinterpretCast<uint8_t>(),w[nz*512],512);
                DataCopy(scale,s[(kb/8)*n+col+localCol],64);
                auto event=pipe.FetchEventID<HardEvent::MTE2_V>();
                SetFlag<HardEvent::MTE2_V>(event); WaitFlag<HardEvent::MTE2_V>(event);
                AscendAntiQuant<int4b_t,bfloat16_t,false>(dst,src,scale,tmp,16,info);
                auto write=pipe.FetchEventID<HardEvent::V_MTE3>();
                SetFlag<HardEvent::V_MTE3>(write); WaitFlag<HardEvent::V_MTE3>(write);
                DataCopyParams copy{16,4,0,(uint16_t)((NTile-64)*2/32)};
                DataCopy(b[(core*2+slot)*k*NTile+kb*16*NTile+localCol],dst,copy);
                PipeBarrier<PIPE_ALL>();
            }
            if(slot==0) Arch::CrossCoreSetFlag<0x2,PIPE_MTE3>(ready0);
            else Arch::CrossCoreSetFlag<0x2,PIPE_MTE3>(ready1);
        }
        for(uint32_t i=iteration>2?iteration-2:0;i<iteration;++i) {
            if(i%2==0) Arch::CrossCoreWaitFlag(free0); else Arch::CrossCoreWaitFlag(free1);
        }
    }
    if ASCEND_IS_AIC {
        uint32_t core=GetBlockIdx();
        using Policy=Gemm::MmadAtlasA2Pingpong<true>;
        using Shape=GemmShape<MTile,NTile,KTile>;
        using L0=GemmShape<MTile,NTile,64>;
        using A=Gemm::GemmType<bfloat16_t,layout::RowMajor>;
        using B=Gemm::GemmType<bfloat16_t,layout::RowMajor>;
        using C=Gemm::GemmType<bfloat16_t,layout::RowMajor>;
        using Block=Gemm::Block::BlockMmad<Policy,Shape,L0,A,B,C>;
        Arch::Resource<Arch::AtlasA2> resource;
        Block matmul(resource);
        GlobalTensor<bfloat16_t> a,b,c;
        a.SetGlobalBuffer((__gm__ bfloat16_t*)x);
        b.SetGlobalBuffer((__gm__ bfloat16_t*)staging);
        c.SetGlobalBuffer((__gm__ bfloat16_t*)output);
        uint32_t iteration=0;
        for(uint32_t col=core*NTile;col<n;col+=20*NTile,++iteration) {
            uint32_t slot=iteration%2;
            if(slot==0) Arch::CrossCoreWaitFlag(ready0); else Arch::CrossCoreWaitFlag(ready1);
            uint32_t cols=n-col<NTile?n-col:NTile;
            for(uint32_t row=0;row<m;row+=MTile) {
                uint32_t rows=m-row<MTile?m-row:MTile;
                matmul(a[row*k],layout::RowMajor(m,k),b[(core*2+slot)*k*NTile],
                    layout::RowMajor(k,NTile),c[row*n+col],layout::RowMajor(m,n),GemmCoord{rows,cols,k});
            }
            PipeBarrier<PIPE_ALL>();
            if(slot==0) Arch::CrossCoreSetFlag<0x2,PIPE_FIX>(free0);
            else Arch::CrossCoreSetFlag<0x2,PIPE_FIX>(free1);
        }
    }
}

template<int NT,int KT>
void launch_nz(void* stream,void* x,void* w,void* s,void* staging,void* y,
               uint32_t m,uint32_t n,uint32_t k) {
    if(m<=16)
        pipeline_nz_gemm<16,NT,KT><<<20,nullptr,stream>>>((GM_ADDR)x,(GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)staging,(GM_ADDR)y,m,n,k);
    else
        pipeline_nz_gemm<128,NT,KT><<<20,nullptr,stream>>>((GM_ADDR)x,(GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)staging,(GM_ADDR)y,m,n,k);
}

extern "C" void w4a16_nz_pipeline(void* stream,void* x,void* w,void* s,void* staging,
    void* y,uint32_t m,uint32_t n,uint32_t k,uint32_t ntile,uint32_t ktile) {
    if(ntile==64 && ktile==256) launch_nz<64,256>(stream,x,w,s,staging,y,m,n,k);
    else if(ntile==64 && ktile==512) launch_nz<64,512>(stream,x,w,s,staging,y,m,n,k);
    else if(ntile==128 && ktile==256) launch_nz<128,256>(stream,x,w,s,staging,y,m,n,k);
    else if(ntile==128 && ktile==512) launch_nz<128,512>(stream,x,w,s,staging,y,m,n,k);
}
