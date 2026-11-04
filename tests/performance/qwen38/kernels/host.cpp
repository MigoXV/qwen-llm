// Framework-aware submission bridge for the experimental NZ kernel.
#include <torch/library.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include "qwen_kernel_build.h"
#include <atomic>
#include <mutex>
#include <ATen/ATen.h>
#include <acl/acl.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

extern "C" void w4a16_nz_pipeline(void*,void*,void*,void*,void*,void*,
    uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);

at::Tensor nz_mm(const at::Tensor& x,const at::Tensor& weight,const at::Tensor& scale,
    const at::Tensor& staging,int64_t ntile,int64_t ktile) {
    TORCH_CHECK(x.dim()==2 && scale.dim()==2,"NZ operator requires matrix inputs");
    int64_t m=x.size(0),k=x.size(1),n=scale.size(1);
    TORCH_CHECK(x.scalar_type()==at::kBFloat16 && scale.scalar_type()==at::kBFloat16 &&
        staging.scalar_type()==at::kBFloat16 && weight.scalar_type()==at::kInt,"invalid W4A16 dtype");
    TORCH_CHECK(x.device()==weight.device() && x.device()==scale.device() && x.device()==staging.device(),"device mismatch");
    TORCH_CHECK(x.is_contiguous() && scale.is_contiguous() && staging.is_contiguous(),"ND inputs must be contiguous");
    TORCH_CHECK(k%128==0 && n%128==0 && weight.size(0)==k && weight.size(1)==n/8 && scale.size(0)==k/128,"invalid NZ W4A16 shape");
    TORCH_CHECK((ntile==64 || ntile==128) && (ktile==256 || ktile==512),"uncompiled tiling");
    TORCH_CHECK(staging.numel()>=40*ntile*k && staging.numel()*2<=1024LL*1024*1024,"invalid staging capacity");
    TORCH_CHECK(at_npu::native::get_npu_format(weight)==29,"packed weight must retain NZ storage");
    TORCH_CHECK(at_npu::native::get_npu_format(x)==2 && at_npu::native::get_npu_format(scale)==2 &&
        at_npu::native::get_npu_format(staging)==2,"activations/scales/staging must be ND");
    auto y=at::empty({m,n},x.options());
    // Resolving stream() in the queue callback drains the queue that is executing
    // that callback and deadlocks. Capture the raw handle without draining here.
    aclrtStream stream=c10_npu::getCurrentNPUStream(x.get_device()).stream(false);
    at_npu::native::OpCommand::RunOpApi("QwenExperimentalNZW4A16",[=]() -> int {
        w4a16_nz_pipeline(stream,x.data_ptr(),weight.data_ptr(),scale.data_ptr(),
            staging.data_ptr(),y.data_ptr(),m,n,k,ntile,ktile);
        return static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL));
    });
    return y;
}

static std::atomic<int64_t> selected_dispatches{0},native_dispatches{0};
static std::atomic<bool> selection_ready{false};
static std::mutex selection_mutex;
static std::vector<int64_t> selected_table;

int64_t register_selection(std::vector<int64_t> values) {
    std::lock_guard<std::mutex> guard(selection_mutex);
    TORCH_CHECK(!selection_ready.load(),"one immutable selection per worker");
    TORCH_CHECK(!values.empty() && values.size()%5==0,"selection must contain M,N,K,Ntile,Ktile rows");
    for(size_t i=0;i<values.size();i+=5) {
        TORCH_CHECK(values[i]>0 && values[i]<=8192 && values[i+2]%128==0 && values[i+1]%128==0,
                    "invalid selection shape");
        TORCH_CHECK((values[i+3]==64 || values[i+3]==128) && (values[i+4]==256 || values[i+4]==512),
                    "invalid selection tile");
    }
    selected_table=std::move(values);
    selection_ready.store(true,std::memory_order_release);
    return 0;
}

at::Tensor nz_selected(const at::Tensor& x,const at::Tensor& weight,const at::Tensor& scale,
    const at::Tensor& staging,int64_t selection_id) {
    TORCH_CHECK(selection_id==0 && selection_ready.load(std::memory_order_acquire),"selection is not registered");
    for(size_t i=0;i<selected_table.size();i+=5) {
        const int64_t* row=selected_table.data()+i;
        if(x.size(0)==row[0] && scale.size(1)==row[1] && x.size(1)==row[2]) {
            selected_dispatches.fetch_add(1,std::memory_order_relaxed);
            return nz_mm(x,weight,scale,staging,row[3],row[4]);
        }
    }
    native_dispatches.fetch_add(1,std::memory_order_relaxed);
    return at_npu::native::custom_ops::npu_weight_quant_batchmatmul(x,weight,scale,
        std::nullopt,std::nullopt,std::nullopt,std::nullopt,128);
}

TORCH_LIBRARY(qwen_w4,m) {
    m.def("nz_mm(Tensor x, Tensor weight, Tensor scale, Tensor(a!) staging, int ntile, int ktile) -> Tensor");
    m.def("nz_selected(Tensor x, Tensor weight, Tensor scale, Tensor(a!) staging, int selection_id) -> Tensor");
    m.def("register_selection(int[] values) -> int",&register_selection);
    m.def("kernel_build_id() -> str",[](){return std::string(QWEN_RAW_KERNEL_SHA256);});
    m.def("kernel_library_path() -> str",[](){return std::string(QWEN_RAW_KERNEL_PATH);});
    m.def("dispatch_counts() -> int[]",[](){return std::vector<int64_t>{selected_dispatches.load(),native_dispatches.load()};});
}
TORCH_LIBRARY_IMPL(qwen_w4,PrivateUse1,m) {
    m.impl("nz_mm",&nz_mm);
    m.impl("nz_selected",&nz_selected);
}
