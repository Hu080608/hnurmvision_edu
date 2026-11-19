#!/usr/bin/env bash
set -eo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SIM_ROOT="$REPO_ROOT/simulation"
source /opt/ros/humble/setup.bash
source "$SIM_ROOT/install_interfaces/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-89}

# ONNX Runtime 的 CUDA provider 是 dlopen 加载的，其依赖只认 LD_LIBRARY_PATH（不吃 RPATH）。
# torch 装的是 CUDA 13，而 ORT 1.23 需要 CUDA 12 运行库，故此处按需补上；没有则跳过。
for _cuda12_dir in "${HOME}/train/ort-cu12/lib" "${HOME}/ort-cu12/lib"; do
  if [ -f "${_cuda12_dir}/libcublas.so.12" ]; then
    export LD_LIBRARY_PATH="${_cuda12_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    break
  fi
done
[ -n "${LD_LIBRARY_PATH:-}" ] && echo "[run] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

cd "$REPO_ROOT"
exec ./build/training configs/simulation.yaml
