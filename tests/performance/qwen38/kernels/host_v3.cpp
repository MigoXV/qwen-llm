#include "host_v2.cpp"
extern "C" void w4a16_v3_run(void*,void*,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
extern "C" void w4a16_v3_stream(void*,void*,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
namespace qwen_v3 {
static at::Tensor panel;
int64_t reserve(const at::Tensor& anchor,int64_t bytes) {
    TORCH_CHECK(!panel.defined() && anchor.device().is_privateuseone() && bytes>0 && bytes<=1024LL*1024*1024,"invalid panel workspace");
    panel=at::empty({(bytes+1)/2},anchor.options().dtype(at::kBFloat16));return bytes;
}
at::Tensor execute(const at::Tensor& x,const at::Tensor& w,const at::Tensor& s,int64_t mt,int64_t nt,int64_t kt,bool streaming) {
    qwen_v2::validate(x,w,s,0);int64_t m=x.size(0),n=s.size(1),k=x.size(1);
    TORCH_CHECK(m>0 && m<=8192 && (mt==16 || mt==128 || mt==256) && (nt==128 || nt==256) && (kt==256 || kt==512),"invalid V3 shape/tile");
    TORCH_CHECK((mt+nt)*kt*4<=512*1024 && mt*nt*4<=128*1024 && !(mt==256 && nt==256),"V3 L1/L0 budget");
    TORCH_CHECK(n%nt==0 && panel.defined() && panel.nbytes()>=40*nt*(streaming?kt:k)*2,"V3 panel workspace too small");
    if(streaming)TORCH_CHECK(k%kt==0 && kt==256 && ((mt==256 && nt==128) || (mt==128 && nt==256)),"unsupported rolling ND profile");
    auto y=at::empty({m,n},x.options());aclrtStream stream=c10_npu::getCurrentNPUStream(x.get_device()).stream(false);
    at_npu::native::OpCommand::RunOpApi("QwenV3Panel",[=]() -> int {
        if(streaming)w4a16_v3_stream(stream,x.data_ptr(),w.data_ptr(),s.data_ptr(),panel.data_ptr(),y.data_ptr(),m,n,k,mt,nt,kt);
        else w4a16_v3_run(stream,x.data_ptr(),w.data_ptr(),s.data_ptr(),panel.data_ptr(),y.data_ptr(),m,n,k,mt,nt,kt);
        return static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL));
    });return y;
}
at::Tensor run(const at::Tensor& x,const at::Tensor& w,const at::Tensor& s,int64_t mt,int64_t nt,int64_t kt){return execute(x,w,s,mt,nt,kt,false);}
at::Tensor rolling(const at::Tensor& x,const at::Tensor& w,const at::Tensor& s,int64_t mt,int64_t nt,int64_t kt){return execute(x,w,s,mt,nt,kt,true);}
}
TORCH_LIBRARY(qwen_w4_v3,m) {
    m.def("reserve(Tensor anchor, int bytes) -> int",&qwen_v3::reserve);
    m.def("rolling(Tensor x, Tensor weight, Tensor scale, int mt, int nt, int kt) -> Tensor");
    m.def("run(Tensor x, Tensor weight, Tensor scale, int mt, int nt, int kt) -> Tensor");
}
TORCH_LIBRARY_IMPL(qwen_w4_v3,PrivateUse1,m) {m.impl("run",&qwen_v3::run);m.impl("rolling",&qwen_v3::rolling);}
