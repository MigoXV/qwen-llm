// Keep V1 ABI while adding private, bounded workspaces for V2.
#include "host.cpp"
#include <memory>
extern "C" void w4a16_v2_run(void*,void*,void*,void*,void*,void*,void*,void*,
    uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
extern "C" void w4a16_v2_dequant(void*,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t);
extern "C" void w4a16_v2_restore(void*,void*,void*,void*,uint32_t,uint32_t);
namespace qwen_v2 {
struct Workspace {
    at::Tensor b,partial,maps,gate,down;
    std::vector<int64_t> configs,profiles;
};
static std::shared_ptr<Workspace> workspace;
static std::mutex mutex;
static std::atomic<int64_t> fast_count{0},fallback_count{0},restore_count{0};
int64_t reserve(at::Tensor maps,at::Tensor gate,at::Tensor down,int64_t bbytes,int64_t pbytes,
                std::vector<int64_t> configs,std::vector<int64_t> profiles) {
    std::lock_guard<std::mutex> guard(mutex);
    TORCH_CHECK(!workspace,"one immutable workspace per worker");
    TORCH_CHECK(maps.device().is_privateuseone() && maps.scalar_type()==at::kInt && maps.numel()==10752,"invalid index maps");
    TORCH_CHECK(configs.size()%6==0 && !configs.empty() && profiles.size()%5==0,"invalid profile/config table");
    TORCH_CHECK(bbytes>0 && pbytes>=0 && bbytes+pbytes+maps.nbytes()+gate.nbytes()+down.nbytes()<=1024LL*1024*1024,"V2 exceeds1GiB workspace budget");
    auto work=std::make_shared<Workspace>();work->maps=maps;work->gate=gate;work->down=down;
    work->configs=std::move(configs);work->profiles=std::move(profiles);
    work->b=at::empty({(bbytes+1)/2},maps.options().dtype(at::kBFloat16));
    work->partial=at::empty({std::max<int64_t>(1,(pbytes+3)/4)},maps.options().dtype(at::kFloat));
    workspace=work;return 0;
}
std::shared_ptr<Workspace> get(int64_t id) {
    TORCH_CHECK(id==0 && workspace,"workspace not registered");return workspace;
}
void validate(const at::Tensor& x,const at::Tensor& w,const at::Tensor& s,int64_t layout) {
    TORCH_CHECK(x.dim()==2 && s.dim()==2 && x.scalar_type()==at::kBFloat16 && s.scalar_type()==at::kBFloat16,"BF16 matrices required");
    TORCH_CHECK(x.device().is_privateuseone() && x.device()==w.device() && x.device()==s.device(),"device mismatch");
    TORCH_CHECK(x.is_contiguous() && s.is_contiguous() && w.is_contiguous(),"contiguous inputs required");
    int64_t k=x.size(1),n=s.size(1);
    TORCH_CHECK(k%128==0 && n%128==0 && s.size(0)==k/128 && w.nbytes()*2==n*k,"invalid W4A16 shapes");
    TORCH_CHECK(at_npu::native::get_npu_format(x)==2 && at_npu::native::get_npu_format(s)==2,"ND activation/scale required");
    TORCH_CHECK((layout==0 && w.scalar_type()==at::kInt && at_npu::native::get_npu_format(w)==29) ||
                (layout==1 && w.scalar_type()==at::kByte && at_npu::native::get_npu_format(w)==2),"weight layout mismatch");
}
at::Tensor scratch(std::shared_ptr<Workspace> work,int64_t n,int64_t k) {
    at::Tensor result;
    if(n==34816 && k==5120)result=work->gate;
    else if(n==5120 && k==17408)result=work->down;
    else TORCH_CHECK(false,"fallback limited to the two verified projections");
    TORCH_CHECK(result.numel()*8==n*k && at_npu::native::get_npu_format(result)==29,"invalid native scratch");
    return result;
}
at::Tensor restore(const at::Tensor& weight,int64_t n,int64_t k,int64_t id) {
    auto work=get(id);auto result=scratch(work,n,k);
    TORCH_CHECK(weight.scalar_type()==at::kByte && weight.nbytes()*2==n*k,"invalid blocked weight");
    aclrtStream stream=c10_npu::getCurrentNPUStream(weight.get_device()).stream(false);
    restore_count.fetch_add(1,std::memory_order_relaxed);
    at_npu::native::OpCommand::RunOpApi("QwenV2RestoreNZ",[=]() -> int {
        w4a16_v2_restore(stream,weight.data_ptr(),result.data_ptr(),work->maps.data_ptr(),n,k);
        return static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL));
    });return result;
}
at::Tensor native(const at::Tensor& x,const at::Tensor& weight,const at::Tensor& scale,int64_t layout,int64_t id) {
    validate(x,weight,scale,layout);fallback_count.fetch_add(1,std::memory_order_relaxed);
    auto w=layout==1?restore(weight,scale.size(1),x.size(1),id):weight;
    return at_npu::native::custom_ops::npu_weight_quant_batchmatmul(x,w,scale,
        std::nullopt,std::nullopt,std::nullopt,std::nullopt,128);
}
at::Tensor run(const at::Tensor& x,const at::Tensor& weight,const at::Tensor& scale,int64_t configuration,int64_t layout,int64_t id) {
    validate(x,weight,scale,layout);auto work=get(id);
    TORCH_CHECK(configuration>=0 && configuration*6+6<=work->configs.size(),"configuration not registered");
    const int64_t* c=work->configs.data()+configuration*6;
    int64_t nt=c[0],kt=c[1],mt=c[2],depth=c[3],split=c[4],family=c[5];
    int64_t m=x.size(0),k=x.size(1),n=scale.size(1);
    TORCH_CHECK((nt==64 || nt==128) && (kt==128 || kt==256 || kt==512) &&
        (mt==16 || mt==64 || mt==128 || mt==256) && (depth==2 || depth==3) &&
        (split==1 || split==2 || split==4 || split==8) && (family==0 || family==1),"invalid configuration");
    TORCH_CHECK((mt+nt)*kt*4<=512*1024 && mt*64*4<=64*1024 && mt*nt*4<=128*1024,"L1/L0 budget exceeded");
    TORCH_CHECK(m>0 && m<=8192 && k%(split*128)==0 && (split==1 || (m<=117 && family==1)),"unsupported split/shape");
    int64_t bbytes=family==0?n*k*2:20*depth*nt*kt*2;
    TORCH_CHECK(work->b.nbytes()>=bbytes && (split==1 || work->partial.nbytes()>=split*m*n*4),"workspace too small");
    auto y=at::empty({m,n},x.options());
    aclrtStream stream=c10_npu::getCurrentNPUStream(x.get_device()).stream(false);
    fast_count.fetch_add(1,std::memory_order_relaxed);
    at_npu::native::OpCommand::RunOpApi("QwenV2W4A16",[=]() -> int {
        w4a16_v2_run(stream,x.data_ptr(),weight.data_ptr(),scale.data_ptr(),work->b.data_ptr(),
            work->partial.data_ptr(),y.data_ptr(),work->maps.data_ptr(),m,n,k,nt,kt,mt,depth,split,family,layout);
        return static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL));
    });return y;
}
at::Tensor selected(const at::Tensor& x,const at::Tensor& w,const at::Tensor& s,int64_t layout,int64_t id) {
    auto work=get(id);
    for(size_t i=0;i<work->profiles.size();i+=5) {
        const int64_t* p=work->profiles.data()+i;
        if(x.size(0)==p[0] && s.size(1)==p[1] && x.size(1)==p[2] && layout==p[3])return run(x,w,s,p[4],layout,id);
    }return native(x,w,s,layout,id);
}
at::Tensor dequant(const at::Tensor& weight,const at::Tensor& scales,int64_t layout,int64_t id) {
    auto work=get(id);int64_t n=scales.size(1),k=scales.size(0)*128;
    TORCH_CHECK(work->b.nbytes()>=n*k*2,"dequant workspace too small");
    aclrtStream stream=c10_npu::getCurrentNPUStream(weight.get_device()).stream(false);
    at_npu::native::OpCommand::RunOpApi("QwenV2Dequant",[=]() -> int {
        w4a16_v2_dequant(stream,weight.data_ptr(),scales.data_ptr(),work->b.data_ptr(),work->maps.data_ptr(),n,k,layout);
        return static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL));
    });return work->b;
}
at::Tensor gemm(const at::Tensor& x,const at::Tensor& scale,int64_t configuration,int64_t id) {
    auto work=get(id);TORCH_CHECK(configuration>=0 && configuration*6+6<=work->configs.size(),"invalid configuration");
    const int64_t* c=work->configs.data()+configuration*6;
    int64_t m=x.size(0),k=x.size(1),n=scale.size(1);
    TORCH_CHECK(work->b.nbytes()>=n*k*2,"GEMM component needs initialized full B workspace");
    auto y=at::empty({m,n},x.options());aclrtStream stream=c10_npu::getCurrentNPUStream(x.get_device()).stream(false);
    at_npu::native::OpCommand::RunOpApi("QwenV2GemmComponent",[=]() -> int {
        w4a16_v2_run(stream,x.data_ptr(),nullptr,nullptr,work->b.data_ptr(),work->partial.data_ptr(),y.data_ptr(),work->maps.data_ptr(),m,n,k,c[0],c[1],c[2],2,1,2,0);
        return static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL));
    });return y;
}
}
TORCH_LIBRARY(qwen_w4_v2,m) {
    m.def("reserve(Tensor maps, Tensor gate, Tensor down, int bbytes, int pbytes, int[] configs, int[] profiles) -> int",&qwen_v2::reserve);
    m.def("counts() -> int[]",[](){return std::vector<int64_t>{qwen_v2::fast_count.load(),qwen_v2::fallback_count.load(),qwen_v2::restore_count.load()};});
    m.def("run(Tensor x, Tensor weight, Tensor scale, int configuration, int layout, int workspace) -> Tensor");
    m.def("selected(Tensor x, Tensor weight, Tensor scale, int layout, int workspace) -> Tensor");
    m.def("native(Tensor x, Tensor weight, Tensor scale, int layout, int workspace) -> Tensor");
    m.def("restore(Tensor weight, int n, int k, int workspace) -> Tensor");
    m.def("dequant(Tensor weight, Tensor scales, int layout, int workspace) -> Tensor");
    m.def("gemm(Tensor x, Tensor scale, int configuration, int workspace) -> Tensor");
}
TORCH_LIBRARY_IMPL(qwen_w4_v2,PrivateUse1,m) {
    m.impl("run",&qwen_v2::run);m.impl("selected",&qwen_v2::selected);m.impl("native",&qwen_v2::native);
    m.impl("restore",&qwen_v2::restore);m.impl("dequant",&qwen_v2::dequant);
    m.impl("gemm",&qwen_v2::gemm);
}
