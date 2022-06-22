#!/bin/bash

gpu_name="T4"
env_name="HalfCheetah-v2"
max_population_size=40
batch_size=256

for method in "vectorized" "sequential" "parallel"
do
    for num_steps_at_once in 1 50
    do
        if [ "${num_steps_at_once}" == "1" ]
        then
            num_iterations=1000
        else
            num_iterations=100
        fi

        for agent in "sac" "td3"
        do
            for library in "torch" "jax"
            do
                for poulation_size in $(seq 1 $max_population_size)
                do
                    echo "Running with population size = ${poulation_size}"
                    td3_flag=""
                    if [ "${agent}" == "td3" ]
                    then
                        td3_flag="--use-td3"
                    fi

                    output_filename="runtime_${gpu_name}_${env_name}_${library}_${agent}_${method}_numsteps${num_steps_at_once}_batch${batch_size}.csv"
                    output_filepath="/app/fastpbrl/${output_filename}"
                    make exp COMMAND="python3 -m timing_scripts.time_train_step_${library} \
                        -o ${output_filepath} \
                        --env-name ${env_name} \
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
done
