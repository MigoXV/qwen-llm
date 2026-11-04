// V3: vectorize the native packed NZ into ND B panels, once per full-K panel.
// Cube ND->zN copy performs the layout transform; no UB Gather or GM roundtrip
// per K slice. A full-K B panel is reused for all M tiles assigned to the core.
#include "w4a16_v2.cpp"

struct VectorNdV3 {
    TPipe pipe;
    TBuf<TPosition::VECCALC> packedBuf,scaleBuf,halfBuf,floatBuf,outBuf,scaleFloatBuf;
    GlobalTensor<uint8_t> weight;
    GlobalTensor<bfloat16_t> scales,stage;
    uint32_t n,k;
    __aicore__ inline VectorNdV3(GM_ADDR w,GM_ADDR s,GM_ADDR b,uint32_t n_,uint32_t k_):n(n_),k(k_) {
        pipe.InitBuffer(packedBuf,4096);pipe.InitBuffer(scaleBuf,128);
        pipe.InitBuffer(halfBuf,16384);pipe.InitBuffer(floatBuf,32768);
        pipe.InitBuffer(outBuf,16384);pipe.InitBuffer(scaleFloatBuf,256);
        weight.SetGlobalBuffer((__gm__ uint8_t*)w);scales.SetGlobalBuffer((__gm__ bfloat16_t*)s);
        stage.SetGlobalBuffer((__gm__ bfloat16_t*)b);
    }
    __aicore__ inline void Panel(uint32_t col,uint32_t nt,uint32_t sub,uint32_t slot,uint32_t firstK=0,uint32_t spanK=0) {
        if(spanK==0)spanK=k;
        auto packed=packedBuf.Get<uint8_t>();auto scale=scaleBuf.Get<bfloat16_t>();
        auto h=halfBuf.Get<half>();auto f=floatBuf.Get<float>();auto out=outBuf.Get<bfloat16_t>();
        auto sf=scaleFloatBuf.Get<float>();
        for(uint32_t localN=sub*64;localN<nt;localN+=128) {
            for(uint32_t row=0;row<spanK;row+=128) {
                DataCopy(packed,weight[((col+localN)/64*k+firstK+row)*32],4096);
                DataCopy(scale,scales[((firstK+row)/128)*n+col+localN],64);
                auto r=pipe.FetchEventID<HardEvent::MTE2_V>();SetFlag<HardEvent::MTE2_V>(r);WaitFlag<HardEvent::MTE2_V>(r);
                Cast(h,packed.ReinterpretCast<int4b_t>(),RoundMode::CAST_NONE,8192);PipeBarrier<PIPE_V>();
                Cast(f,h,RoundMode::CAST_NONE,8192);
                Cast(sf,scale,RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
                BinaryRepeatParams broadcast{1,1,1,8,8,0};
                Mul(f,f,sf,64,128,broadcast);PipeBarrier<PIPE_V>();
                Cast(out,f,RoundMode::CAST_RINT,8192);
                auto v=pipe.FetchEventID<HardEvent::V_MTE3>();SetFlag<HardEvent::V_MTE3>(v);WaitFlag<HardEvent::V_MTE3>(v);
                DataCopyParams cp{128,4,0,(uint16_t)((nt-64)/16)};
                DataCopy(stage[slot*spanK*nt+row*nt+localN],out,cp);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
};

template<int MT,int NT,int KT>
__global__ __aicore__ void panel_v3(GM_ADDR x,GM_ADDR w,GM_ADDR scales,GM_ADDR stage,GM_ADDR y,
                                    uint32_t m,uint32_t n,uint32_t k) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV {
        uint32_t core=GetBlockIdx()/GetSubBlockNum();VectorNdV3 vector(w,scales,stage,n,k);
        uint32_t index=0;
        for(uint32_t col=core*NT;col<n;col+=20*NT,++index) {
            uint32_t slot=index%2;
            if(index>=2){Arch::CrossCoreFlag free(slot+2);Arch::CrossCoreWaitFlag(free);}
            vector.Panel(col,NT,GetSubBlockIdx(),core*2+slot);
            Arch::CrossCoreFlag ready(slot);Arch::CrossCoreSetFlag<0x2,PIPE_MTE3>(ready);
        }
        for(uint32_t i=index>2?index-2:0;i<index;++i){Arch::CrossCoreFlag free(i%2+2);Arch::CrossCoreWaitFlag(free);}
    }
    if ASCEND_IS_AIC {
        using A=Gemm::GemmType<bfloat16_t,layout::RowMajor>;
        using B=Gemm::GemmType<bfloat16_t,layout::RowMajor>;
        using C=Gemm::GemmType<bfloat16_t,layout::RowMajor>;
        using Block=Gemm::Block::BlockMmad<Gemm::MmadAtlasA2Pingpong<true>,GemmShape<MT,NT,KT>,GemmShape<MT,NT,64>,A,B,C>;
        Arch::Resource<Arch::AtlasA2> resource;Block block(resource);
        GlobalTensor<bfloat16_t>a,b,c;a.SetGlobalBuffer((__gm__ bfloat16_t*)x);
        b.SetGlobalBuffer((__gm__ bfloat16_t*)stage);c.SetGlobalBuffer((__gm__ bfloat16_t*)y);
        uint32_t core=GetBlockIdx(),index=0;
        for(uint32_t col=core*NT;col<n;col+=20*NT,++index) {
            uint32_t slot=index%2;Arch::CrossCoreFlag ready(slot);Arch::CrossCoreWaitFlag(ready);
            for(uint32_t row=0;row<m;row+=MT)
                block(a[row*k],layout::RowMajor(m,k),b[(core*2+slot)*k*NT],layout::RowMajor(k,NT),
                      c[row*n+col],layout::RowMajor(m,n),GemmCoord{min((uint32_t)MT,m-row),NT,k});
            PipeBarrier<PIPE_ALL>();Arch::CrossCoreFlag free(slot+2);Arch::CrossCoreSetFlag<0x2,PIPE_FIX>(free);
        }
    }
}

template<int M,int N>
void panel_k_dispatch(void*st,void*x,void*w,void*s,void*b,void*y,uint32_t m,uint32_t n,uint32_t k,uint32_t kt) {
    if(kt==256)panel_v3<M,N,256><<<20,nullptr,st>>>((GM_ADDR)x,(GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)b,(GM_ADDR)y,m,n,k);
    else {if constexpr ((M+N)*512*4<=512*1024) {if(kt==512)panel_v3<M,N,512><<<20,nullptr,st>>>((GM_ADDR)x,(GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)b,(GM_ADDR)y,m,n,k);}}
}
extern "C" void w4a16_v3_run(void*st,void*x,void*w,void*s,void*b,void*y,uint32_t m,uint32_t n,uint32_t k,uint32_t mt,uint32_t nt,uint32_t kt) {
    if(mt==16){if(nt==128)panel_k_dispatch<16,128>(st,x,w,s,b,y,m,n,k,kt);else panel_k_dispatch<16,256>(st,x,w,s,b,y,m,n,k,kt);}
    else if(mt==128){if(nt==128)panel_k_dispatch<128,128>(st,x,w,s,b,y,m,n,k,kt);else panel_k_dispatch<128,256>(st,x,w,s,b,y,m,n,k,kt);}
    else if(mt==256 && nt==128)panel_k_dispatch<256,128>(st,x,w,s,b,y,m,n,k,kt);
}

// Rolling ND panels: hardware ND->zN on Cube, 256-row L0C tile for reuse.
template<int NT,int KT>
struct RollingNdCopyV3 {
    using Base=Gemm::Tile::CopyGmToL1<Arch::AtlasA2,Gemm::GemmType<bfloat16_t,layout::RowMajor>>;
    using LayoutSrc=layout::RowMajor;using LayoutDst=typename Base::LayoutDst;
    GlobalTensor<bfloat16_t> base;uint32_t index=0;
    __aicore__ inline void operator()(LocalTensor<bfloat16_t> dst,GlobalTensor<bfloat16_t> ignored,LayoutDst dl,LayoutSrc sl) {
        uint32_t slot=index++%2;Arch::CrossCoreFlag ready(slot),free(slot+2);Arch::CrossCoreWaitFlag(ready);
        Base copy;copy(dst,base[slot*NT*KT],dl,layout::RowMajor(sl.shape(0),NT));
        Arch::CrossCoreSetFlag<0x2,PIPE_MTE2>(free);
    }
};

template<int MT,int NT,int KT>
__global__ __aicore__ void stream_nd_v3(GM_ADDR x,GM_ADDR w,GM_ADDR scales,GM_ADDR stage,GM_ADDR y,uint32_t m,uint32_t n,uint32_t k) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    uint32_t core=GetBlockIdx();if ASCEND_IS_AIV {core/=GetSubBlockNum();}
    uint32_t nn=n/NT,nm=(m+MT-1)/MT;
    if ASCEND_IS_AIV {
        VectorNdV3 vector(w,scales,stage,n,k);
        for(uint32_t task=core;task<nn*nm;task+=20) {
            uint32_t col=task%nn*NT,iterations=k/KT;
            for(uint32_t i=0;i<iterations;++i) {
                uint32_t slot=i%2;if(i>=2){Arch::CrossCoreFlag free(slot+2);Arch::CrossCoreWaitFlag(free);}
                vector.Panel(col,NT,GetSubBlockIdx(),core*2+slot,i*KT,KT);
                Arch::CrossCoreFlag ready(slot);Arch::CrossCoreSetFlag<0x2,PIPE_MTE3>(ready);
            }
            for(uint32_t i=iterations>2?iterations-2:0;i<iterations;++i){Arch::CrossCoreFlag free(i%2+2);Arch::CrossCoreWaitFlag(free);}
        }
    }
    if ASCEND_IS_AIC {
        using A=Gemm::GemmType<bfloat16_t,layout::RowMajor>;using B=A;using C=A;
        struct Copies:Gemm::Tile::TileCopy<Arch::AtlasA2,A,B,C>{using CopyGmToL1B=RollingNdCopyV3<NT,KT>;};
        using Parent=Gemm::Block::BlockMmad<Gemm::MmadAtlasA2Pingpong<true>,GemmShape<MT,NT,KT>,GemmShape<MT,NT,64>,A,B,C,void,Copies>;
        struct Block:Parent {
            __aicore__ inline Block(Arch::Resource<Arch::AtlasA2>& r):Parent(r){}
            __aicore__ inline void Reset(GlobalTensor<bfloat16_t> base){this->copyGmToL1B.base=base;this->copyGmToL1B.index=0;}
        };
        Arch::Resource<Arch::AtlasA2> resource;Block block(resource);
        GlobalTensor<bfloat16_t>a,b,c;a.SetGlobalBuffer((__gm__ bfloat16_t*)x);b.SetGlobalBuffer((__gm__ bfloat16_t*)stage);c.SetGlobalBuffer((__gm__ bfloat16_t*)y);
        for(uint32_t task=core;task<nn*nm;task+=20) {
            uint32_t col=task%nn*NT,row=task/nn*MT;block.Reset(b[core*2*NT*KT]);
            block(a[row*k],layout::RowMajor(m,k),b,layout::RowMajor(k,NT),c[row*n+col],layout::RowMajor(m,n),GemmCoord{min((uint32_t)MT,m-row),NT,k});
            PipeBarrier<PIPE_ALL>();
        }
    }
}
extern "C" void w4a16_v3_stream(void*st,void*x,void*w,void*s,void*b,void*y,uint32_t m,uint32_t n,uint32_t k,uint32_t mt,uint32_t nt,uint32_t kt) {
    if(mt==256 && nt==128) {
        if(kt==256)stream_nd_v3<256,128,256><<<20,nullptr,st>>>((GM_ADDR)x,(GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)b,(GM_ADDR)y,m,n,k);
        
    } else if(mt==128 && nt==256 && kt==256)stream_nd_v3<128,256,256><<<20,nullptr,st>>>((GM_ADDR)x,(GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)b,(GM_ADDR)y,m,n,k);
}
