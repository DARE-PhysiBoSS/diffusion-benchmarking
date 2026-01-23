srun   --cpus-per-task=1 --qos=gp_debug --account=bsc08  build/diffuse --problem example-problems/problem.json --params example-problems/params.json --benchmark --alg lstmfpai --double
srun: job 35318135 queued and waiting for resources
srun: job 35318135 has been allocated resources
algorithm,precision,dims,iterations,s,nx,ny,nz,cores_division,xs_tile_size,init_time,time,std_dev
lstmfpai,D,3,1,8,100,100,100,[1,1,1],256,28,22825.5,4160.916347
lstmfpai,D,3,1,8,100,100,100,[1,1,1],256,36,25113.9,292.130262
lstmfpai,D,3,1,8,100,100,100,[1,1,1],256,32,27295.1,3325.92151
lstmfpai,D,3,1,8,100,100,100,[1,1,1],256,32,26163.4,216.9079067
lstmfpai,D,3,1,8,100,100,100,[1,1,1],256,32,26249,195.6532647



srun   --cpus-per-task=1 --qos=gp_debug --account=bsc08  build/diffuse --problem example-problems/problem.json --params example-problems/params.json --benchmark --alg lstcta   --double
srun: job 35318094 queued and waiting for resources
srun: job 35318094 has been allocated resources
algorithm,precision,dims,iterations,s,nx,ny,nz,cores_division,xs_tile_size,init_time,time,std_dev
lstcta,D,3,1,8,100,100,100,[1,1,1],256,418,24766.8,284.7868677
lstcta,D,3,1,8,100,100,100,[1,1,1],256,454,24880.5,611.8160263
lstcta,D,3,1,8,100,100,100,[1,1,1],256,437,25169.4,1390.106413
lstcta,D,3,1,8,100,100,100,[1,1,1],256,457,24369.5,455.998739
lstcta,D,3,1,8,100,100,100,[1,1,1],256,438,24420.6,238.3762572

srun   --cpus-per-task=1 --qos=gp_debug --account=bsc08  build/diffuse --problem example-problems/problem.json --params example-problems/params.json --benchmark --alg ref   --double
srun: job 35318115 queued and waiting for resources
srun: job 35318115 has been allocated resources
algorithm,precision,dims,iterations,s,nx,ny,nz,cores_division,xs_tile_size,init_time,time,std_dev
ref,D,3,1,8,100,100,100,[1,1,1],256,6,47806.3,355.1413944
ref,D,3,1,8,100,100,100,[1,1,1],256,7,47176.1,273.8307689
ref,D,3,1,8,100,100,100,[1,1,1],256,6,46929.4,208.620325
ref,D,3,1,8,100,100,100,[1,1,1],256,6,46746.6,169.3258397
ref,D,3,1,8,100,100,100,[1,1,1],256,7,46450.1,180.9687542



srun --constraint=perfparanoid   --cpus-per-task=1 --qos=gp_debug --account=bsc08  build/diffuse --problem example-problems/problem.json --params example-problems/params.json --profile --alg lstmfpai --double

"cycles": "70705102",
"real_time_nsec": "35351576",
"PAPI_TOT_INS": "73306485",
"PAPI_VEC_INS": "15635200"

MP_NUM_THREADS=1 srun --constraint=perfparanoid   --cpus-per-task=1 --qos=gp_debug --account=bsc08  build/diffuse --problem example-problems/problem.json --params example-problems/params.json --profile --alg lstct --double

"cycles": "81205906",
"real_time_nsec": "40601832",
"PAPI_TOT_INS": "156527563",
"PAPI_VEC_INS": "29760000"
