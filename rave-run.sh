OMP_NUM_THREADS=1 RAVE_PRINT_REPORT=1 rave ./build/riscrel/diffuse --problem example-problems/problem.json --params example-problems/params.json --alg lstmfpai --run --double
------------------- REPORT -------------------
Region #0: Event -1 (Global), Value 1 (-), Rank 0, Thread -1
        Moved bytes (Total): 12944148831
                Moved bytes (scalar): 18836831 (0.15 %)
                Moved bytes (vector): 12925312000 (99.85 %)
        tot_instr: 80404731
                scalar_instr: 33304698 (41.42 %)
                vsetvl_instr: 12962417 (16.12 %)
                vector_instr: 34137616 (42.46 %)
                        SEW 8 vector_instr: 7432000 (21.77 %)
                                avg_VL: 2048.00 elements
                                Arith: 1648800 (22.19 %)
                                        FP: 0 (0.00 %)
                                        INT: 1648800 (100.00 %)
                                Mem: 5783200 (77.81 %)
                                        unit: 5783200 (100.00 %)
                                        strided: 0 (0.00 %)
                                        indexed: 0 (0.00 %)
                                Mask: 0 (0.00 %)
                                Other: 0 (0.00 %)
                        SEW 16 vector_instr: 0 (0.00 %)
                        SEW 32 vector_instr: 0 (0.00 %)
                        SEW 64 vector_instr: 26705616 (78.23 %)
                                avg_VL: 223.90 elements
                                Arith: 4604800 (17.24 %)
                                        FP: 4596800 (99.83 %)
                                        INT: 8000 (0.17 %)
                                Mem: 4687200 (17.55 %)
                                        unit: 4687200 (100.00 %)
                                        strided: 0 (0.00 %)
                                        indexed: 0 (0.00 %)
                                Mask: 6400 (0.02 %)
                                Other: 17407216 (65.18 %)
------------------------------------------------


OMP_NUM_THREADS=1 RAVE_PRINT_REPORT=1 rave ./build/riscrel/diffuse --problem example-problems/problem.json --params example-problems/params.json --alg lstct --run --double
------------------- REPORT -------------------
Region #0: Event -1 (Global), Value 1 (-), Rank 0, Thread -1
        Moved bytes (Total): 483487163
                Moved bytes (scalar): 479647163 (99.21 %)
                Moved bytes (vector): 3840000 (0.79 %)
        tot_instr: 1142324755
                scalar_instr: 1142320755 (100.00 %)
                vsetvl_instr: 800 (0.00 %)
                vector_instr: 3200 (0.00 %)
                        SEW 8 vector_instr: 0 (0.00 %)
                        SEW 16 vector_instr: 0 (0.00 %)
                        SEW 32 vector_instr: 0 (0.00 %)
                        SEW 64 vector_instr: 3200 (100.00 %)
                                avg_VL: 200.00 elements
                                Arith: 800 (25.00 %)
                                        FP: 800 (100.00 %)
                                        INT: 0 (0.00 %)
                                Mem: 2400 (75.00 %)
                                        unit: 2400 (100.00 %)
                                        strided: 0 (0.00 %)
                                        indexed: 0 (0.00 %)
                                Mask: 0 (0.00 %)
                                Other: 0 (0.00 %)
------------------------------------------------


OMP_NUM_THREADS=1 RAVE_PRINT_REPORT=1 rave ./build/riscrel/diffuse --problem example-problems/problem.json --params example-problems/params.json --alg ref --run --double     
------------------- REPORT -------------------
Region #0: Event -1 (Global), Value 1 (-), Rank 0, Thread -1
        Moved bytes (Total): 1327027983
                Moved bytes (scalar): 54067983 (4.07 %)
                Moved bytes (vector): 1272960000 (95.93 %)
        tot_instr: 361393527
                scalar_instr: 327583527 (90.64 %)
                vsetvl_instr: 3990000 (1.10 %)
                vector_instr: 29820000 (8.25 %)
                        SEW 8 vector_instr: 0 (0.00 %)
                        SEW 16 vector_instr: 0 (0.00 %)
                        SEW 32 vector_instr: 0 (0.00 %)
                        SEW 64 vector_instr: 29820000 (100.00 %)
                                avg_VL: 8.00 elements
                                Arith: 9930000 (33.30 %)
                                        FP: 9930000 (100.00 %)
                                        INT: 0 (0.00 %)
                                Mem: 19890000 (66.70 %)
                                        unit: 19890000 (100.00 %)
                                        strided: 0 (0.00 %)
                                        indexed: 0 (0.00 %)
                                Mask: 0 (0.00 %)
                                Other: 0 (0.00 %)
------------------------------------------------