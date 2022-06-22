import logging
import os
import time
from multiprocessing import Barrier, Queue
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from fastpbrl.dqn.core import DQNHyperParameters
from fastpbrl.torch.dqn import DQN
from fastpbrl.torch.dqn_pbt import DQNPBT
from fastpbrl.torch.networks import (
    DQNAtariCritic,
    VectorizedDQNAtariCritic,
    torch_load_on_device,
)
from fastpbrl.types import Transition
from timing_scripts.dqn_common import (
    MeasureConfig,
    Method,
    get_parser_args,
    measure_runtimes_parallel,
)


def generate_training_batch(config: MeasureConfig, population_size: int) -> Transition:
    batch_size = config.batch_size
    num_steps_at_once = config.num_steps_at_once
    num_actions = config.num_actions

    obs_dim = (
        num_steps_at_once * batch_size,
        config.num_frame_stack * population_size,
        *config.image_shape,
    )

    transition = Transition(
        observation=np.random.rand(*obs_dim).astype(np.float32),
        action=np.random.randint(
            low=0,
            high=num_actions,
            size=(population_size, num_steps_at_once * batch_size),
        ).astype(np.int64),
        reward=np.random.rand(population_size, num_steps_at_once * batch_size).astype(
            np.float32
        ),
        done=np.zeros((population_size, num_steps_at_once * batch_size)).astype(
            np.float32
        ),
        next_observation=np.random.rand(*obs_dim).astype(np.float32),
    )
    return transition


def create_agent(config: MeasureConfig) -> Union[DQN, DQNPBT]:
    device = torch.device("cuda:0" if config.use_gpu else "cpu")

    # Define the torch networks that will be used
    def create_critic_network() -> torch.nn.Module:
        critic_network_args = {
            "image_shape": config.image_shape,
            "num_channels": config.num_frame_stack,
            "num_actions": config.num_actions,
        }

        if config.method == Method.VECTORIZED:
            critic_network_args["population_size"] = config.population_size
            return VectorizedDQNAtariCritic(**critic_network_args)
        return DQNAtariCritic(**critic_network_args)

    dqn_args = {
        "critic_network": create_critic_network().to(device),
        "target_critic_network": create_critic_network().to(device),
        "hyperparams": DQNHyperParameters(),
    }
    if config.method == Method.VECTORIZED:
        dqn_args["population_size"] = config.population_size
        return DQNPBT(**dqn_args)
    return DQN(**dqn_args)


def measure_runtimes_torch_vectorized(
    config: MeasureConfig,
) -> Tuple[float, float]:

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if config.use_gpu else "cpu")

    # Pre-load a batch of transitions
    transition_batch = generate_training_batch(
        config,
        population_size=config.population_size,
    )
    transition_batch = torch_load_on_device(transition_batch, device)
    transition_batch = transition_batch._replace(
        observation=transition_batch.observation.to(
            device,
            memory_format=torch.channels_last,
        ),
        next_observation=transition_batch.next_observation.to(
            device,
            memory_format=torch.channels_last,
        ),
    )

    agent = create_agent(config)

    def measure_train_step_time() -> float:
        if config.use_gpu:
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        agent.train_step(transition_batch, num_steps=config.num_steps_at_once)
        if config.use_gpu:
            torch.cuda.synchronize(device)
        return time.perf_counter() - start_time

    all_runtimes = [measure_train_step_time() for _ in range(config.num_iterations)]

    avg_runtime = np.mean(all_runtimes[2:]) / config.num_steps_at_once
    compilation_time = all_runtimes[0] / config.num_steps_at_once
    return (avg_runtime, compilation_time)


def measure_runtimes_torch_sequential(
    config: MeasureConfig,
    barrier_init: Optional[Barrier] = None,
    all_barrier_train: Optional[List[Barrier]] = None,
) -> Tuple[float, float]:

    torch.set_num_threads(1)

    if barrier_init is not None:
        barrier_init.wait()

    device = torch.device("cuda:0" if config.use_gpu else "cpu")

    # Pre-load a batch of transitions
    transition_batch = generate_training_batch(
        config,
        population_size=1,
    )
    transition_batch = transition_batch._replace(
        action=np.squeeze(transition_batch.action, axis=0),
        reward=np.squeeze(transition_batch.reward, axis=0),
        done=np.squeeze(transition_batch.done, axis=0),
    )  # Remove the population_size dimension
    transition_batch = torch_load_on_device(transition_batch, device)
    transition_batch = transition_batch._replace(
        observation=transition_batch.observation.to(
            device, memory_format=torch.channels_last
        ),
        next_observation=transition_batch.next_observation.to(
            device, memory_format=torch.channels_last
        ),
    )

    # Define the agents that will be used
    all_agents = [create_agent(config) for _ in range(config.population_size)]

    def measure_train_step_time() -> float:
        if config.use_gpu:
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        for agent in all_agents:
            agent.train_step(transition_batch, num_steps=config.num_steps_at_once)
        if config.use_gpu:
            torch.cuda.synchronize(device)
        return time.perf_counter() - start_time

    compilation_time = measure_train_step_time() / config.num_steps_at_once

    if all_barrier_train is not None:
        for barrier in all_barrier_train:
            barrier.wait()

    all_runtimes = [measure_train_step_time() for _ in range(config.num_iterations)]

    avg_runtime = np.mean(all_runtimes[2:]) / config.num_steps_at_once

    return (avg_runtime, compilation_time)


def measure_one_runtime_torch_parallel(
    config: MeasureConfig,
    barrier_init: Barrier,
    all_barrier_train: List[Barrier],
    measurements_queue: Queue,
):
    try:
        measurement = measure_runtimes_torch_sequential(
            config=config,
            barrier_init=barrier_init,
            all_barrier_train=all_barrier_train,
        )
    except RuntimeError:
        logging.exception(
            "An error was raised when measuring runtimes with torch parallel"
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
        avg_runtime, compilation_time = measure_runtimes_torch_vectorized(config)
    elif config.method == Method.PARALLEL:
        avg_runtime, compilation_time = measure_runtimes_parallel(
            config, measure_runtime_func=measure_one_runtime_torch_parallel
        )
    else:
        avg_runtime, compilation_time = measure_runtimes_torch_sequential(config)

    with open(output_filepath, "a") as output_file:
        output_file.write(
            f"{config.population_size},{avg_runtime},{compilation_time}\n"
        )
