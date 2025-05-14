

# commercial RISC-V board arriesgado-11
module load llvm/EPI-development papi/risc-v openBLAS/ubuntu/0.3.20_gcc10.3.0 cmake
cmake -DCMAKE_PREFIX_PATH=/apps/riscv/ubuntu/papi/RISC-V-PAPI -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -B build -S .
cmake --build build --target diffuse
./build/diffuse --problem example-problems/50x50x50x1.json --alg lstcta --validate

# arriesgado-11 with RVV 1.0 (running results in SIGILL)
module load llvm/EPI-development papi/risc-v openBLAS/ubuntu/0.3.20_gcc10.3.0 cmake
cmake -DCMAKE_PREFIX_PATH=/apps/riscv/ubuntu/papi/RISC-V-PAPI -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS='-O3 -ffast-math -mepi -mllvm -combiner-store-merging=0 -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize -mllvm -vectorizer-use-vp-strided-load-store -mcpu=avispado -mllvm -disable-loop-idiom-memcpy -mllvm -disable-loop-idiom-memset -Rpass-missed=loop-vectorize -Xclang -target-feature -Xclang +does-not-implement-vszext -Xclang -target-feature -Xclang +does-not-implement-tu -mllvm -riscv-uleb128-reloc=0 -fno-slp-vectorize' -B build -S .
cmake --build build --target diffuse
./build/diffuse --problem example-problems/50x50x50x1.json --alg lstcta --validate

# arriesgado-11 with RVV 1.0 and SDV Tracing (running results in SIGILL)
module load llvm/EPI-development papi/risc-v openBLAS/ubuntu/0.3.20_gcc10.3.0 cmake sdv_trace
cmake -DCMAKE_PREFIX_PATH=/apps/riscv/ubuntu/papi/RISC-V-PAPI -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -ffast-math -mepi -mllvm -combiner-store-merging=0 -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize -mllvm -vectorizer-use-vp-strided-load-store -mcpu=avispado -mllvm -disable-loop-idiom-memcpy -mllvm -disable-loop-idiom-memset -Rpass-missed=loop-vectorize -Xclang -target-feature -Xclang +does-not-implement-vszext -Xclang -target-feature -Xclang +does-not-implement-tu -mllvm -riscv-uleb128-reloc=0 -fno-slp-vectorize ${SDV_TRACE_INCL}" -B build -S .
cmake --build build --target diffuse
./build/diffuse --problem example-problems/50x50x50x1.json --alg lstcta --validate

# on synth-hca
module load rave sdv_trace
trace_rave_1_0 ./build/diffuse --problem example-problems/50x50x50x1.json --alg lstcta --validate