#!/bin/bash

# Install RISC-V cross-compiler
.devcontainer/install_rvv_crosscompiler.sh

# Install LLVM
sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

# Install plotting R dependencies
sudo R -e "install.packages(c('ggplot2', 'cowplot', 'sitools', 'viridis', 'dplyr'), repos='https://cloud.r-project.org')"
