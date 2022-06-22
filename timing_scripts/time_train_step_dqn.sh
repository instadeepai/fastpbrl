#!/bin/bash

gpu_name="T4"
max_population_size=40
batch_size=32

for method in "vectorized" "sequential" "parallel"
do
    for num_steps_at_once in 1 10
    do
        if [ "${num_steps_at_once}" == "1" ]
        then
            num_iterations=1000
        else
            num_iterations=100
        fi

        for library in "torch" "jax"
        do
            for poulation_size in $(seq 1 $max_population_size)
            do
                echo "Running with population size = ${poulation_size}"

                output_filename="runtime_${gpu_name}_Atari_${library}_dqn_${method}_numsteps${num_steps_at_once}_batch${batch_size}.csv"
                output_filepath="/app/fastpbrl/${output_filename}"
                make exp COMMAND="python3 -m timing_scripts.time_train_step_${library}_dqn \
                    -o ${output_filepath} \
                    --population-size ${poulation_size} \
                    --num-steps-at-once ${num_steps_at_once} \
                    --num-iterations ${num_iterations} \
                    --method ${method} \
                    --batch-size ${batch_size} \
                    --use-gpu ${td3_flag}"

                last_population_size_entry=$(tail -1 ${output_filename} | cut -d "," -f 1)
                if [ "${last_population_size_entry}" != "${poulation_size}" ]
                then
                    break
                fi
            done
        done
    done
done
