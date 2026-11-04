"""单独构建带TorchNPU任务队列语义的实验桥，不安装到运行环境。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
import torch_npu
from torch.utils.cpp_extension import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',choices=['host.cpp','host_v2.cpp','host_v3.cpp'],default='host.cpp')
    args=p.parse_args()
    if not args.library.is_file() or args.output.exists():
        raise ValueError('动态库必须存在，输出目录必须不存在')
    args.output.mkdir(parents=True)
    source=args.output/'source.cpp'
    source.write_bytes((Path(__file__).with_name('kernels')/args.source).read_bytes())
    for included in Path(__file__).with_name('kernels').glob('host*.cpp'):
        (args.output/included.name).write_bytes(included.read_bytes())
    (args.output/'qwen_kernel_build.h').write_text(
        '#define QWEN_RAW_KERNEL_SHA256 '+json.dumps(hashlib.sha256(args.library.read_bytes()).hexdigest())+'\n'
        '#define QWEN_RAW_KERNEL_PATH '+json.dumps(str(args.library.resolve()))+'\n')
    npu_root=Path(torch_npu.__file__).parent
    result=load(name='qwen_w4_host',sources=[str(source.resolve())],
        extra_include_paths=[str(args.output.resolve()),str(npu_root/'include'),str(npu_root/'include/third_party/acl/inc')],
        extra_ldflags=[str(args.library.resolve()),'-Wl,-rpath,'+str(args.library.resolve().parent),
                       '-L'+str(Path(os.environ['ASCEND_HOME_PATH'])/'lib64'),'-lascendcl',
                       '-L'+str(npu_root/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu_root/'lib')],
        extra_cflags=['-O2'],build_directory=str(args.output.resolve()),
        with_cuda=False,is_python_module=False,verbose=True)
    (args.output/'build.json').write_text(json.dumps({
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'raw_library':str(args.library.resolve()),
        'raw_library_sha256':hashlib.sha256(args.library.read_bytes()).hexdigest(),
        'host_library':str(result),
        'host_library_sha256':hashlib.sha256(Path(result).read_bytes()).hexdigest(),
        'torch':torch.__version__,'torch_npu':torch_npu.__version__,
        'cann_root':str(Path(os.environ['ASCEND_HOME_PATH']).resolve())},indent=2)+'\n')
    print(result)


if __name__=='__main__':main()
