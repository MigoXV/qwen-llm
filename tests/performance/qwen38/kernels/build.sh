#!/usr/bin/env bash
set -eo pipefail
kernel_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$kernel_root/../../../.." && pwd)"
source "$project_root/scripts/env_ascend_291.sh"
output="${1:?提供不存在的绝对构建目录}"
[[ "$output" == /* && ! -e "$output" ]] || exit 2
mkdir -p "$output"
source_name="${2:-w4a16.cpp}"
[[ "$source_name" == w4a16.cpp || "$source_name" == w4a16_v2.cpp || "$source_name" == w4a16_v3.cpp ]] || exit 2
mkdir "$output/source"
cp "$kernel_root"/*.cpp "$output/source/"
cp "$output/source/$source_name" "$output/source.cpp"
catlass_root="${QWEN_CATLASS_ROOT:-$QWEN_RUNTIME_ROOT/src/ascend018/csrc/third_party/catlass}"
compiler="$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler/bin/bisheng"
asc_root="$ASCEND_HOME_PATH/aarch64-linux/asc"
"$compiler" -O2 -std=c++17 -xcce -fPIC --cce-aicore-arch=dav-c220 \
    -I"$output/source" -I"$asc_root/include" -I"$asc_root/include/basic_api" -I"$asc_root/include/adv_api" \
    -I"$asc_root" -I"$ASCEND_HOME_PATH/include" \
    -I"$catlass_root/include" -I"$catlass_root/examples/shared_lib/src" \
    -I"$catlass_root/examples/shared_lib/src/common" \
    -c "$output/source.cpp" -o "$output/w4a16.o"
"$compiler" --cce-fatobj-link --shared "$output/w4a16.o" \
    -L"$ASCEND_HOME_PATH/lib64" -lruntime -lascendcl -lstdc++ -lm -ldl \
    -o "$output/libqwen_w4a16.so"
sha256sum "$output/source.cpp" "$output/libqwen_w4a16.so" > "$output/sha256.txt"
