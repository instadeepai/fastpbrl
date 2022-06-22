import logging
import os
import time
from multiprocessing import Barrier, Queue
from typing import List, Optional, Tuple

import jax
import numpy as np
import tree

from fastpbrl.agents.sac import SAC
from fastpbrl.agents.sac_pbt import SACPBT
from fastpbrl.agents.td3 import TD3
from fastpbrl.agents.td3_pbt import TD3PBT
from fastpbrl.td3.core import TD3HyperParams
from fastpbrl.types import TrainingState
from timing_scripts.td3_sac_common import (
    MeasureConfig,
    Method,
    default_sac_hyperparams,
    generate_training_batch,
    get_parser_args,
    measure_runtimes_parallel,
)


def measure_runtimes_jax_vectorized(
    config: MeasureConfig,
) -> Tuple[float, float]:
    device = jax.devices(backend="gpu" if config.use_gpu else "cpu")[0]

    # Use the same values of the hyperparameters for all members of the population
    hyperparams = tree.map_structure(
        lambda x: np.array([x for _ in range(config.population_size)]),
        TD3HyperParams()
        if config.use_td3
        else default_sac_hyperparams(config.env_name),
    )

    # Load the hyperparameters on the jax device
    hyperparams = jax.device_put(hyperparams, device)
    jax.tree_map(lambda x: x.block_until_ready(), hyperparams)

    # Define the agent that will be used
    agent = (TD3PBT if config.use_td3 else SACPBT)(
        gym_env_name=config.env_name,
        seed=0,
        population_size=config.population_size,
        hidden_layer_sizes=config.hidden_layer_sizes,
    )

    # Pre-load a batch of transitions
    transition_batch = generate_training_batch(config, config.population_size)
    transition_batch = jax.device_put(transition_batch, device)
    jax.tree_map(lambda x: x.block_until_ready(), transition_batch)

    def measure_train_step_time(state: TrainingState) -> Tuple[TrainingState, float]:
        start_time = time.perf_counter()
        state = agent.update_step(
            state, hyperparams, transition_batch, config.num_steps_at_once
        )
        state = jax.tree_map(lambda x: x.block_until_ready(), state)
        return state, time.perf_counter() - start_time

    # Initialize the training state and measure the compilation time
    training_state = agent.make_initial_training_state()
    training_state = jax.device_put(training_state, device)
    jax.tree_map(lambda x: x.block_until_ready(), training_state)
    training_state, compilation_time = measure_train_step_time(training_state)

    # Measure train-step runtimes over multiple iterations
    all_runtimes = []
    for _ in range(config.num_iterations):
        training_state, runtime = measure_train_step_time(training_state)
        all_runtimes.append(runtime)

    return (np.mean(all_runtimes[2:]) / config.num_steps_at_once, compilation_time)


def measure_runtimes_jax_sequential(
    config: MeasureConfig,
    barrier_init: Optional[Barrier] = None,
    all_barrier_train: Optional[List[Barrier]] = None,
) -> Tuple[float, float]:

    if barrier_init is not None:
        barrier_init.wait()

    device = jax.devices(backend="gpu" if config.use_gpu else "cpu")[0]

    # Use the same values of the hyperparameters for all members of the population
    all_hyperparams = [
        TD3HyperParams() if config.use_td3 else default_sac_hyperparams(config.env_name)
        for _ in range(config.population_size)
    ]

    # Load the hyperparameters on the jax device
    all_hyperparams = tree.map_structure(
        lambda x: jax.device_put(x, device).block_until_ready(), all_hyperparams
    )

    # Define the agents that will be used
    all_agent = [
        (TD3 if config.use_td3 else SAC)(
            gym_env_name=config.env_name,
            seed=0,
            hidden_layer_sizes=config.hidden_layer_sizes,
        )
        for _ in range(config.population_size)
    ]

    # Pre-load a batch of transitions
    transition_batch = generate_training_batch(config, population_size=1)
    transition_batch = tree.map_structure(
        lambda x: np.squeeze(x, axis=0), transition_batch
    )  # Remove the population_size dimension
    transition_batch = jax.device_put(transition_batch, device)
    jax.tree_map(lambda x: x.block_until_ready(), transition_batch)

    def measure_train_step_time(
        all_state: List[TrainingState],
    ) -> Tuple[List[TrainingState], float]:
        start_time = time.perf_counter()
        all_state = [
            agent.update_step(
                state, hyperparams, transition_batch, config.num_steps_at_once
            )
            for agent, state, hyperparams in zip(all_agent, all_state, all_hyperparams)
        ]
        all_state = jax.tree_map(lambda x: x.block_until_ready(), all_state)
        return all_state, time.perf_counter() - start_time

    # Initialize the training states and measure the compilation time
    all_training_state = [agent.make_initial_training_state() for agent in all_agent]
    all_training_state = [
        jax.device_put(training_state, device) for training_state in all_training_state
    ]
    for training_state in all_training_state:
        jax.tree_map(lambda x: x.block_until_ready(), training_state)
    all_training_state, compilation_time = measure_train_step_time(all_training_state)

    if all_barrier_train is not None:
        for barrier in all_barrier_train:
            barrier.wait()

    # Measure train-step runtimes over multiple iterations
    all_runtimes = []
    for _ in range(config.num_iterations):
        all_training_state, runtime = measure_train_step_time(all_training_state)
        all_runtimes.append(runtime)

    avg_runtime = np.mean(all_runtimes[2:]) / config.num_steps_at_once
    return (avg_runtime, compilation_time)


def measure_one_runtime_jax_parallel(
    config: MeasureConfig,
    barrier_init: Barrier,
    all_barrier_train: List[Barrier],
    measurements_queue: Queue,
):
    try:
        measurement = measure_runtimes_jax_sequential(
            config=config,
            barrier_init=barrier_init,
            all_barrier_train=all_barrier_train,
        )
    except RuntimeError:
        logging.exception(
            "An error was raised when measuring runtimes with jax parallel"
        )

        measurement = None

    measurements_queue.put(measurement)


if __name__ == "__main__":
    args = get_parser_args()
    config = MeasureConfig.from_args(args)

    output_filepath = args.o
    if not os.path.isfile(output_filepath):
        with open(output_filepath, "w") as output_file:
            output_file.write("population_size,runtime_per_step,compilation_time\n")

    if config.method == Method.VECTORIZED:
        avg_runtime, compilation_time = measure_runtimes_jax_vectorized(config)
    elif config.method == Method.PARALLEL:
        avg_runtime, compilation_time = measure_runtimes_parallel(
            config, measure_runtime_func=measure_one_runtime_jax_parallel
        )
    else:
        avg_runtime, compilation_time = measure_runtimes_jax_sequential(config)

    with open(output_filepath, "a") as output_file:
        output_file.write(
            f"{config.population_size},{avg_runtime},{compilation_time}\n"
        )
