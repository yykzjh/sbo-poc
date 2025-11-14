# install dependencies
uv sync
source ./.venv/bin/activate

pushd third_party

# build nvshmem
pushd nvshmem
mkdir -p build install
NVSHMEM_INSTALL=$PWD/install

NVSHMEM_SHMEM_SUPPORT=0 \
NVSHMEM_UCX_SUPPORT=0 \
NVSHMEM_USE_NCCL=0 \
NVSHMEM_MPI_SUPPORT=0 \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_PMIX_SUPPORT=0 \
NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
NVSHMEM_USE_GDRCOPY=1 \
cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=$NVSHMEM_INSTALL -DCMAKE_CUDA_ARCHITECTURES=90

pushd build
make -j$(nproc)
make install

export NVSHMEM_DIR=$NVSHMEM_INSTALL
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
popd # build
popd # nvshmem

# build deep_ep_sbo
pushd deep_ep_sbo
rm -rf build dist
rm -rf *.egg-info
if ! grep -q "extra_link_args.extend(\['-lcudadevrt'\])" setup.py; then
    python3 << 'EOF'
with open('setup.py', 'r') as f:
    lines = f.readlines()

new_line = "        extra_link_args.extend(['-lcudadevrt'])\n"
lines.insert(71, new_line)

with open('setup.py', 'w') as f:
    f.writelines(lines)
EOF
fi
TORCH_CUDA_ARCH_LIST="9.0" python setup.py bdist_wheel
uv pip install dist/*.whl --force-reinstall
popd # deep_ep_sbo

# build deep_gemm_sbo
pushd deep_gemm_sbo
rm -rf build dist
rm -rf *.egg-info
python setup.py bdist_wheel
uv pip install dist/*.whl --force-reinstall
popd # deep_gemm_sbo

popd # third_party
