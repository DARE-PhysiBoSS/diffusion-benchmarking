#!/bin/bash

set -x

if [ -d ".rave" ]; then
    exit 0
fi

git clone https://repo.hca.bsc.es/gitlab/pvizcaino/rave.git .rave
cd .rave

./install_qemu.sh 1_0
./install_compiler.sh 1_0
./install_rave.sh 1_0
./install_sysroot.sh 
./install_parallel.sh
./install_gdb.sh 1_0

cd ..

cp .devcontainer/rave-env .rave/build/llvm-cross/llvm-EPI-development-toolchain-cross/bin
chmod +x .rave/build/llvm-cross/llvm-EPI-development-toolchain-cross/bin/rave-env
