// V2: group128 vector batches, rolling K slots and persistent CATLASS L0C.
// TileCopy/TileMmad implementations remain in the unmodified CATLASS install.
#include "w4a16.cpp"
using namespace Catlass;

struct VectorDequantV2 {
    TPipe pipe;
    TBuf<TPosition::VECCALC> inputBuf,rawScaleBuf,outputBuf;
    TEventID sourceReady[2],sourceFree[2],outputReady[2],outputFree[2];
    uint32_t loadIndex=0,computeIndex=0;
    TBuf<TPosition::VECCALC> ndBuf,scaleBuf,indexBuf,tmpBuf;
    GlobalTensor<uint8_t> weight;
    GlobalTensor<bfloat16_t> scales,stage;
    uint32_t n,k,layout;
    __aicore__ inline VectorDequantV2(GM_ADDR w,GM_ADDR s,GM_ADDR b,GM_ADDR maps,
                                     uint32_t n_,uint32_t k_,uint32_t layout_) : n(n_),k(k_),layout(layout_) {
        pipe.InitBuffer(inputBuf,8192);pipe.InitBuffer(rawScaleBuf,256);pipe.InitBuffer(outputBuf,32768);
        for(int i=0;i<2;++i) {
            sourceReady[i]=pipe.AllocEventID<HardEvent::MTE2_V>();sourceFree[i]=pipe.AllocEventID<HardEvent::V_MTE2>();
            outputReady[i]=pipe.AllocEventID<HardEvent::V_MTE3>();outputFree[i]=pipe.AllocEventID<HardEvent::MTE3_V>();
            SetFlag<HardEvent::V_MTE2>(sourceFree[i]);SetFlag<HardEvent::MTE3_V>(outputFree[i]);
        }
        pipe.InitBuffer(ndBuf,16384); pipe.InitBuffer(scaleBuf,4096);
        pipe.InitBuffer(indexBuf,40960); pipe.InitBuffer(tmpBuf,65536);
        weight.SetGlobalBuffer((__gm__ uint8_t*)w); scales.SetGlobalBuffer((__gm__ bfloat16_t*)s);
        stage.SetGlobalBuffer((__gm__ bfloat16_t*)b);
        GlobalTensor<uint32_t> gm; gm.SetGlobalBuffer((__gm__ uint32_t*)maps);
        DataCopy(indexBuf.Get<uint32_t>(),gm,10240);
        auto e=pipe.FetchEventID<HardEvent::MTE2_V>(); SetFlag<HardEvent::MTE2_V>(e);WaitFlag<HardEvent::MTE2_V>(e);
    }
    __aicore__ inline void Load(uint32_t row,uint32_t col,uint32_t rows) {
        uint32_t slot=loadIndex++%2;
        WaitFlag<HardEvent::V_MTE2>(sourceFree[slot]);
        auto bundle=inputBuf.Get<uint8_t>()[slot*4096];auto scale=rawScaleBuf.Get<bfloat16_t>()[slot*64];
        uint32_t offset=layout==0 ? ((col/64)*(k/16)+row/16)*1024
            : ((col/64)*(k/128)+row/128)*8192+(row%128)*64;
        // On A2 DataCopy's count overload uses sizeof(int4b_t)==1, not nibble
        // units. Copy explicit bytes to avoid overwriting the next UB slot.
        DataCopy(bundle,weight[offset/2],rows*32);
        DataCopy(scale,scales[(row/128)*n+col],64);
        SetFlag<HardEvent::MTE2_V>(sourceReady[slot]);
    }
    __aicore__ inline void ComputeStore(uint32_t rows,uint32_t offset,uint32_t strideN) {
        uint32_t slot=computeIndex++%2;
        WaitFlag<HardEvent::MTE2_V>(sourceReady[slot]);WaitFlag<HardEvent::MTE3_V>(outputFree[slot]);
        auto src=inputBuf.Get<int4b_t>()[slot*8192];auto scale=rawScaleBuf.Get<bfloat16_t>()[slot*64];
        auto result=outputBuf.Get<bfloat16_t>()[slot*8192];
        // AntiQuant uses LocalTensor::GetSize, not only the explicit shape.
        // Queue slices keep their allocation size unless it is narrowed here.
        src.SetSize(rows*64);scale.SetSize(64);result.SetSize(rows*64);
        auto tmp=tmpBuf.Get<uint8_t>(); auto indices=indexBuf.Get<uint32_t>();
        auto halfValues=tmp.ReinterpretCast<half>();
        auto values=tmp[8192].ReinterpretCast<float>();
        auto floatScale=tmp[32768].ReinterpretCast<float>();
        if(layout==0) {
            auto nd=ndBuf.Get<bfloat16_t>();nd.SetSize(4096);
            Cast(floatScale,scale,RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            for(uint32_t first=0;first<rows;first+=64) {
                auto input=src[first*64];input.SetSize(4096);
                Cast(halfValues,input,RoundMode::CAST_NONE,4096);PipeBarrier<PIPE_V>();
                Cast(values,halfValues,RoundMode::CAST_NONE,4096);PipeBarrier<PIPE_V>();
                BinaryRepeatParams broadcast{1,1,1,8,8,0};
                Mul(values,values,floatScale,64,64,broadcast);PipeBarrier<PIPE_V>();
                Cast(nd,values,RoundMode::CAST_RINT,4096);
                PipeBarrier<PIPE_V>();
                Gather(result[first*64].ReinterpretCast<uint16_t>(),nd.ReinterpretCast<uint16_t>(),indices,0,4096);
                PipeBarrier<PIPE_V>();
            }
        } else {
            auto expanded=scaleBuf.Get<bfloat16_t>();
            Gather(expanded.ReinterpretCast<uint16_t>(),scale.ReinterpretCast<uint16_t>(),indices[8192],0,2048);
            PipeBarrier<PIPE_V>();
            Cast(floatScale,expanded,RoundMode::CAST_NONE,2048);PipeBarrier<PIPE_V>();
            for(uint32_t i=0;i<rows*64;i+=2048) {
                auto target=result[i];auto input=src[i];target.SetSize(2048);input.SetSize(2048);
                Cast(halfValues,input,RoundMode::CAST_NONE,2048);PipeBarrier<PIPE_V>();
                Cast(values,halfValues,RoundMode::CAST_NONE,2048);PipeBarrier<PIPE_V>();
                Mul(values,values,floatScale,2048);PipeBarrier<PIPE_V>();
                Cast(target,values,RoundMode::CAST_RINT,2048);
                PipeBarrier<PIPE_V>();
            }
        }
        SetFlag<HardEvent::V_MTE2>(sourceFree[slot]);
        SetFlag<HardEvent::V_MTE3>(outputReady[slot]);WaitFlag<HardEvent::V_MTE3>(outputReady[slot]);
        if(strideN<=128) {
            DataCopyParams cp{(uint16_t)(rows/16),64,0,(uint16_t)(strideN-64)};
            DataCopy(stage[offset],result,cp);
        } else {
            for(uint32_t r=0;r<rows/16;++r)DataCopy(stage[offset+r*strideN*16],result[r*1024],1024);
        }
        SetFlag<HardEvent::MTE3_V>(outputFree[slot]);
    }
    __aicore__ inline ~VectorDequantV2() {
        for(int i=0;i<2;++i) {
            WaitFlag<HardEvent::V_MTE2>(sourceFree[i]);WaitFlag<HardEvent::MTE3_V>(outputFree[i]);
            pipe.ReleaseEventID<HardEvent::MTE2_V>(sourceReady[i]);pipe.ReleaseEventID<HardEvent::V_MTE2>(sourceFree[i]);
            pipe.ReleaseEventID<HardEvent::V_MTE3>(outputReady[i]);pipe.ReleaseEventID<HardEvent::MTE3_V>(outputFree[i]);
        }
    }
    __aicore__ inline void Tile(uint32_t row,uint32_t col,uint32_t actualK,uint32_t ntile,
                               uint32_t sub,uint32_t offset,uint32_t strideN) {
        uint32_t rows=ntile==64?64:128;
        uint32_t localN=ntile==128?sub*64:0, localK=ntile==64?sub*64:0;
        uint32_t count=actualK/128;
        for(uint32_t i=0;i<min(count,2u);++i) Load(row+i*128+localK,col+localN,rows);
        for(uint32_t i=0;i<count;++i) {
            uint32_t dst=offset+(i*128+localK)/16*strideN*16+localN*16;
            ComputeStore(rows,dst,strideN);
            if(i+2<count)Load(row+(i+2)*128+localK,col+localN,rows);
        }
    }
};

template<int NT,int KT>
struct RollingCopyV2 {
    using Base=Gemm::Tile::CopyGmToL1<Arch::AtlasA2,Gemm::GemmType<bfloat16_t,layout::nZ>>;
    using LayoutSrc=layout::nZ; using LayoutDst=layout::nZ;
    GlobalTensor<bfloat16_t> base;
    uint32_t index=0,depth=2;
    __aicore__ inline void operator()(LocalTensor<bfloat16_t> dst,GlobalTensor<bfloat16_t> ignored,
                                     LayoutDst dl,LayoutSrc sl) {
        uint32_t slot=index++%depth;
        Arch::CrossCoreFlag ready(slot),free(slot+depth);
        Arch::CrossCoreWaitFlag(ready);
        Base copy;
        copy(dst,base[slot*NT*KT],dl,layout::nZ::MakeLayout<bfloat16_t>(sl.orgShape(0),NT));
        // MTE2 flag is delivered only after the B tile has been read into L1.
        Arch::CrossCoreSetFlag<0x2,PIPE_MTE2>(free);
    }
};

template<int MT,int NT,int KT,typename C>
struct RollingBlockV2 {
    using A=Gemm::GemmType<bfloat16_t,layout::RowMajor>;
    using B=Gemm::GemmType<bfloat16_t,layout::nZ>;
    using Out=Gemm::GemmType<C,layout::RowMajor>;
    struct Copies:Gemm::Tile::TileCopy<Arch::AtlasA2,A,B,Out> {using CopyGmToL1B=RollingCopyV2<NT,KT>;};
    using Parent=Gemm::Block::BlockMmad<Gemm::MmadAtlasA2Pingpong<true>,GemmShape<MT,NT,KT>,
        GemmShape<MT,NT,64>,A,B,Out,void,Copies>;
    struct Block:Parent {
        __aicore__ inline Block(Arch::Resource<Arch::AtlasA2>& r):Parent(r){}
        __aicore__ inline void Reset(GlobalTensor<bfloat16_t> stage,uint32_t depth) {
            this->copyGmToL1B.base=stage;this->copyGmToL1B.depth=depth;this->copyGmToL1B.index=0;
        }
    };
};

template<int MT,int NT,int KT,typename Out>
__global__ __aicore__ void stream_v2(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR b,GM_ADDR y,GM_ADDR maps,
                                   uint32_t m,uint32_t n,uint32_t k,uint32_t depth,uint32_t split,uint32_t layoutId) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    uint32_t core=GetBlockIdx();
    if ASCEND_IS_AIV {core/=GetSubBlockNum();}
    uint32_t nm=(m+MT-1)/MT,nn=n/NT,tasks=nm*nn*split,chunk=k/split;
    if ASCEND_IS_AIV {
        VectorDequantV2 vector(w,s,b,maps,n,k,layoutId);
        for(uint32_t task=core;task<tasks;task+=20) {
            uint32_t col=(task%nn)*NT,part=task/(nn*nm),iterations=(chunk+KT-1)/KT;
            for(uint32_t i=0;i<iterations;++i) {
                uint32_t slot=i%depth;
                if(i>=depth) {Arch::CrossCoreFlag free(slot+depth);Arch::CrossCoreWaitFlag(free);}
                vector.Tile(part*chunk+i*KT,col,min((uint32_t)KT,chunk-i*KT),NT,GetSubBlockIdx(),
                            (core*depth+slot)*NT*KT,NT);
                Arch::CrossCoreFlag ready(slot);Arch::CrossCoreSetFlag<0x2,PIPE_MTE3>(ready);
            }
            for(uint32_t i=iterations>depth?iterations-depth:0;i<iterations;++i) {
                Arch::CrossCoreFlag free(i%depth+depth);Arch::CrossCoreWaitFlag(free);
            }
        }
    }
    if ASCEND_IS_AIC {
        Arch::Resource<Arch::AtlasA2> resource;
        typename RollingBlockV2<MT,NT,KT,Out>::Block block(resource);
        GlobalTensor<bfloat16_t> a,stage;
        GlobalTensor<Out> c;
        a.SetGlobalBuffer((__gm__ bfloat16_t*)x);stage.SetGlobalBuffer((__gm__ bfloat16_t*)b);
        c.SetGlobalBuffer((__gm__ Out*)y);
        for(uint32_t task=core;task<tasks;task+=20) {
            uint32_t col=(task%nn)*NT,row=((task/nn)%nm)*MT,part=task/(nn*nm);
            block.Reset(stage[core*depth*NT*KT],depth);
            block(a[row*k+part*chunk],layout::RowMajor(m,k),stage,layout::nZ::MakeLayout<bfloat16_t>(chunk,NT),
                  c[part*m*n+row*n+col],layout::RowMajor(m,n),GemmCoord{min((uint32_t)MT,m-row),NT,chunk});
            PipeBarrier<PIPE_ALL>();
        }
    }
}

__global__ __aicore__ void dequant_v2(GM_ADDR w,GM_ADDR s,GM_ADDR b,GM_ADDR maps,uint32_t n,uint32_t k,uint32_t layoutId) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    VectorDequantV2 vector(w,s,b,maps,n,k,layoutId);
    uint32_t tiles=(n/64)*(k/128);
    for(uint32_t task=GetBlockIdx();task<tiles;task+=GetBlockNum()) {
        uint32_t col=(task/(k/128))*64,row=task%(k/128)*128;
        vector.Load(row,col,128);vector.ComputeStore(128,(row/16)*n*16+col*16,n);
    }
}

template<int MT,int NT,int KT>
__global__ __aicore__ void gemm_nz_v2(GM_ADDR x,GM_ADDR w,GM_ADDR y,uint32_t m,uint32_t n,uint32_t k) {
    using A=Gemm::GemmType<bfloat16_t,layout::RowMajor>;
    using B=Gemm::GemmType<bfloat16_t,layout::nZ>;
    using C=Gemm::GemmType<bfloat16_t,layout::RowMajor>;
    using Block=Gemm::Block::BlockMmad<Gemm::MmadAtlasA2Pingpong<true>,GemmShape<MT,NT,KT>,GemmShape<MT,NT,64>,A,B,C>;
    using Swizzle=Gemm::Block::GemmIdentityBlockSwizzle<3,1>;
    using Kernel=Gemm::Kernel::BasicMatmul<Block,void,Swizzle>;
    typename Kernel::Params params{GemmCoord{m,n,k},x,layout::RowMajor(m,k),w,
        layout::nZ::MakeLayout<bfloat16_t>(k,n),y,layout::RowMajor(m,n)};
    Kernel kernel;kernel(params);
}

template<int MT,int NT,int KT>
void launch_v2(void* stream,void*x,void*w,void*s,void*b,void*partial,void*y,void*maps,
               uint32_t m,uint32_t n,uint32_t k,uint32_t depth,uint32_t split,uint32_t layoutId,uint32_t family) {
    if constexpr ((MT+NT)*KT*4<=512*1024 && MT*64*4<=64*1024 && MT*NT*4<=128*1024) {
        if(family==0 || family==2) {
            if(family==0)dequant_v2<<<40,nullptr,stream>>>((GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)b,(GM_ADDR)maps,n,k,layoutId);
            gemm_nz_v2<MT,NT,KT><<<20,nullptr,stream>>>((GM_ADDR)x,(GM_ADDR)b,(GM_ADDR)y,m,n,k);
        } else if(split==1) {
            stream_v2<MT,NT,KT,bfloat16_t><<<20,nullptr,stream>>>((GM_ADDR)x,(GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)b,(GM_ADDR)y,(GM_ADDR)maps,m,n,k,depth,split,layoutId);
        } else {
            stream_v2<MT,NT,KT,float><<<20,nullptr,stream>>>((GM_ADDR)x,(GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)b,(GM_ADDR)partial,(GM_ADDR)maps,m,n,k,depth,split,layoutId);
            reduce_parts<<<40,nullptr,stream>>>((GM_ADDR)partial,(GM_ADDR)y,m*n,split);
        }
    }
}

#define DISPATCH_K(M,N) if(kt==128)launch_v2<M,N,128>(stream,x,w,s,b,p,y,maps,m,n,k,d,split,layout,family);else if(kt==256)launch_v2<M,N,256>(stream,x,w,s,b,p,y,maps,m,n,k,d,split,layout,family);else if(kt==512)launch_v2<M,N,512>(stream,x,w,s,b,p,y,maps,m,n,k,d,split,layout,family)
#define DISPATCH_N(M) if(nt==64){DISPATCH_K(M,64);}else if(nt==128){DISPATCH_K(M,128);}
extern "C" void w4a16_v2_run(void*stream,void*x,void*w,void*s,void*b,void*p,void*y,void*maps,
    uint32_t m,uint32_t n,uint32_t k,uint32_t nt,uint32_t kt,uint32_t mt,uint32_t d,uint32_t split,uint32_t family,uint32_t layout) {
    if(mt==16){DISPATCH_N(16);}else if(mt==64){DISPATCH_N(64);}else if(mt==128){DISPATCH_N(128);}else if(mt==256){DISPATCH_N(256);}
}

extern "C" void w4a16_v2_dequant(void*stream,void*w,void*s,void*b,void*maps,uint32_t n,uint32_t k,uint32_t layout) {
    dequant_v2<<<40,nullptr,stream>>>((GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)b,(GM_ADDR)maps,n,k,layout);
}

__global__ __aicore__ void restore_nz_v2(GM_ADDR packed,GM_ADDR output,GM_ADDR maps,uint32_t n,uint32_t k) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;TBuf<TPosition::VECCALC> srcBuf,outBuf,tmpBuf,indexBuf;
    pipe.InitBuffer(srcBuf,4096);pipe.InitBuffer(outBuf,1024);pipe.InitBuffer(tmpBuf,3072);pipe.InitBuffer(indexBuf,2048);
    auto src=srcBuf.Get<uint16_t>(),out=outBuf.Get<uint16_t>(),tmp=tmpBuf.Get<uint16_t>();
    auto indices=indexBuf.Get<uint32_t>();GlobalTensor<uint16_t> w,y;GlobalTensor<uint32_t> idx;
    w.SetGlobalBuffer((__gm__ uint16_t*)packed);y.SetGlobalBuffer((__gm__ uint16_t*)output);idx.SetGlobalBuffer((__gm__ uint32_t*)maps);
    DataCopy(indices,idx[10240],512);
    for(uint32_t task=GetBlockIdx();task<(n/64)*(k/128);task+=GetBlockNum()) {
        uint32_t col=task/(k/128),row=task%(k/128);
        DataCopy(src,w[task*2048],2048);
        auto e=pipe.FetchEventID<HardEvent::MTE2_V>();SetFlag<HardEvent::MTE2_V>(e);WaitFlag<HardEvent::MTE2_V>(e);
        for(uint32_t rem=0;rem<4;++rem) {
            Duplicate(out,(uint16_t)0,512);
            Duplicate(tmp[512],(uint16_t)15,512);
            PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<4;++c) {
                Gather(tmp,src,indices,c*8,512);
                PipeBarrier<PIPE_V>();
                ShiftRight(tmp,tmp,(uint16_t)(rem*4),512);PipeBarrier<PIPE_V>();
                And(tmp,tmp,tmp[512],512);PipeBarrier<PIPE_V>();
                ShiftLeft(tmp,tmp,(uint16_t)(c*4),512);PipeBarrier<PIPE_V>();
                Or(out,out,tmp,512);PipeBarrier<PIPE_V>();
            }
            auto v=pipe.FetchEventID<HardEvent::V_MTE3>();SetFlag<HardEvent::V_MTE3>(v);WaitFlag<HardEvent::V_MTE3>(v);
            DataCopyParams cp{32,1,0,3};DataCopy(y[col*k*16+row*2048+rem*16],out,cp);
            PipeBarrier<PIPE_ALL>();
        }
    }
}
extern "C" void w4a16_v2_restore(void*stream,void*w,void*y,void*maps,uint32_t n,uint32_t k) {
    restore_nz_v2<<<40,nullptr,stream>>>((GM_ADDR)w,(GM_ADDR)y,(GM_ADDR)maps,n,k);
}
