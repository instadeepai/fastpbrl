import logging
import os
import time
from multiprocessing import Barrier, Queue
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import tree

from fastpbrl.td3.core import TD3HyperParams
from fastpbrl.torch.networks import (
    ContinuousDeterministicActor,
    ContinuousQNetwork,
    SquashedGaussianActor,
    VectorizedContinuousDeterministicActor,
    VectorizedContinuousQNetwork,
    VectorizedSquashedGaussianActor,
    torch_load_on_device,
)
from fastpbrl.torch.sac import SAC
from fastpbrl.torch.sac_pbt import SACPBT
from fastpbrl.torch.td3 import TD3
from fastpbrl.torch.td3_pbt import TD3PBT
from fastpbrl.utils import make_gym_env
from timing_scripts.td3_sac_common import (
    MeasureConfig,
    Method,
    default_sac_hyperparams,
    generate_training_batch,
    get_parser_args,
    measure_runtimes_parallel,
)


def create_agent(config: MeasureConfig) -> Union[TD3, SAC, TD3PBT, SACPBT]:
    device = torch.device("cuda:0" if config.use_gpu else "cpu")

    # Define the torch networks that will be used
    gym_env = make_gym_env(config.env_name)
    obs_dim = np.prod(gym_env.observation_space.high.shape)
    action_dim = np.prod(gym_env.action_space.high.shape)
    max_action = torch.Tensor(np.array(gym_env.action_space.high)).to(device)
    min_action = torch.Tensor(np.array(gym_env.action_space.low)).to(device)

    def create_critic_network() -> torch.nn.Module:
        critic_network_args = {
            "observation_dim": obs_dim,
            "action_dim": action_dim,
            "layers_dim": list(config.hidden_layer_sizes),
            "activation": "ELU" if config.use_td3 else "ReLU",
            "use_layer_norm": True if config.use_td3 else False,
        }

        if config.method == Method.VECTORIZED:
            critic_network_args["population_size"] = config.population_size
            return VectorizedContinuousQNetwork(**critic_network_args)
        return ContinuousQNetwork(**critic_network_args)

    def create_policy_network() -> torch.nn.Module:
        policy_network_args = {
            "observation_dim": obs_dim,
            "action_dim": action_dim,
            "layers_dim": list(config.hidden_layer_sizes),
            "max_action": max_action,
            "min_action": min_action,
            "activation": "ELU" if config.use_td3 else "ReLU",
            "use_layer_norm": True if config.use_td3 else False,
        }

        if config.method == Method.VECTORIZED:
            policy_network_args["population_size"] = config.population_size

            if config.use_td3:
                return VectorizedContinuousDeterministicActor(**policy_network_args)
            return VectorizedSquashedGaussianActor(**policy_network_args)

        if config.use_td3:
            return ContinuousDeterministicActor(**policy_network_args)
        return SquashedGaussianActor(**policy_network_args)

    if config.use_td3:
        td3_args = {
            "policy_network": create_policy_network().to(device),
            "critic_network": create_critic_network().to(device),
            "target_policy_network": create_policy_network().to(device),
            "target_critic_network": create_critic_network().to(device),
            "hyperparams": TD3HyperParams(),
            "min_action": min_action,
            "max_action": max_action,
        }
        if config.method == Method.VECTORIZED:
            td3_args["population_size"] = config.population_size
            return TD3PBT(**td3_args)
        return TD3(**td3_args)

    sac_args = {
        "policy_network": create_policy_network().to(device),
        "critic_network": create_critic_network().to(device),
        "target_critic_network": create_critic_network().to(device),
        "hyperparams": default_sac_hyperparams(config.env_name),
    }
    if config.method == Method.VECTORIZED:
        sac_args["population_size"] = config.population_size
        return SACPBT(**sac_args)
    return SAC(**sac_args)


def measure_runtimes_torch_vectorized(
    config: MeasureConfig,
) -> Tuple[float, float]:

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if config.use_gpu else "cpu")

    # Pre-load a batch of transitions
    transition_batch = generate_training_batch(
        config, population_size=config.population_size
    )
    # Swap the first two axes, as both TD3PBT and SACPBT
    # expect the first dimension to be the number of train steps
    # to carry out at once
    transition_batch = tree.map_structure(
        lambda np_array: np.swapaxes(np_array, 0, 1), transition_batch
    )
    transition_batch = torch_load_on_device(transition_batch, device)

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
    transition_batch = generate_training_batch(config, population_size=1)
    transition_batch = tree.map_structure(
        lambda x: np.squeeze(x, axis=0), transition_batch
    )  # Remove the population_size dimension
    transition_batch = torch_load_on_device(transition_batch, device)

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
