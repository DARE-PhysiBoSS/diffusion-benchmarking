#!/bin/bash

# Define the list of algorithms to test
algorithms=("biofvm" "lstcma" "lstcta" "lstcstai" "lstmtai" "lstmdtai" "lstmdtfai")

params_file="example-problems/params.json"

out_dir="output"

# Write the initial problem JSON to a file
problem_file="build/problem.json"
cat > "$problem_file" <<EOF
{
    "dims": 3,
    "dx": 20,
    "dy": 20,
    "dz": 20,
    "nx": 100,
    "ny": 100,
    "nz": 100,
    "substrates_count": 1,
    "iterations": 100,
    "dt": 0.01,
    "diffusion_coefficients": 10000,
    "decay_rates": 0.01,
    "initial_conditions": 1000,
    "gaussian_pulse": false
}
EOF

# Define the sets of values to test
n_values=($(seq 25 25 300))
substrates_values=(1 8 16 32 64)
counters=("PAPI_FP_OPS,PAPI_FP_INS" "PAPI_VEC_INS,PAPI_TOT_INS" "PAPI_L3_TCM,PAPI_L2_TCM,PAPI_L1_TCM" "PAPI_L2_DCM,PAPI_L1_DCM" "PAPI_BR_MSP")

for alg in "${algorithms[@]}"; do
    for n in "${n_values[@]}"; do
        for substrates in "${substrates_values[@]}"; do
            for counter in "${counters[@]}"; do
                # Update the JSON file in-place
                jq \
                    --argjson nx "$n" \
                    --argjson ny "$n" \
                    --argjson nz "$n" \
                    --argjson substrates "$substrates" \
                    '.nx=$nx | .ny=$ny | .nz=$nz | .substrates_count=$substrates' \
                    "$problem_file" > "$problem_file.tmp" && mv "$problem_file.tmp" "$problem_file"

                for dtype in "s" "d"; do
                    logdir="${out_dir}/profile_${alg}_${dtype}_${n}x${substrates}"

                    if [ -d "$logdir" ]; then
                        echo "Skipping $logdir (already exists)"
                        continue
                    fi

                    mkdir $logdir

                    export PAPI_EVENTS="${counter}"
                    export PAPI_OUTPUT_DIRECTORY="${logdir}"

                    if [ "$dtype" = "single" ]; then
                        build/diffuse --problem "$problem_file" --params "$params_file" --alg "$alg" --profile
                    else
                        build/diffuse --problem "$problem_file" --params "$params_file" --alg "$alg" --profile --double
                    fi
                done
            done
        done
    done
done
