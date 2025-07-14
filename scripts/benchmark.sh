#!/bin/bash

# Define the list of algorithms to test
algorithms=("biofvm" "lstcma" "lstcta" "lstcstai" "lstmtai" "lstmdtai" "lstmdtfai")

params_file="example-problems/params.json"

log_dir="output"

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

for alg in "${algorithms[@]}"; do
    for n in "${n_values[@]}"; do
        for substrates in "${substrates_values[@]}"; do
            # Update the JSON file in-place
            jq \
                --argjson nx "$n" \
                --argjson ny "$n" \
                --argjson nz "$n" \
                --argjson substrates "$substrates" \
                '.nx=$nx | .ny=$ny | .nz=$nz | .substrates_count=$substrates' \
                "$problem_file" > "$problem_file.tmp" && mv "$problem_file.tmp" "$problem_file"

            for dtype in "s" "d"; do
                logfile="${log_dir}/benchmark_${alg}_${dtype}_${n}x${substrates}.out"

                if [ -f "$logfile" ]; then
                    echo "Skipping $logfile (already exists)"
                    continue
                fi

                if [ "$dtype" = "single" ]; then
                    build/diffuse --problem "$problem_file" --params "$params_file" --alg "$alg" --benchmark | tee -a "$logfile"
                else
                    build/diffuse --problem "$problem_file" --params "$params_file" --alg "$alg" --benchmark --double | tee -a "$logfile"
                fi
            done
        done
    done
done
