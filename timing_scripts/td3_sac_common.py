import argparse
import copy
import os
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Barrier, Process, Queue, set_start_method
from typing import Callable, List, Tuple

import numpy as np

from fastpbrl.sac.core import SACHyperParams
from fastpbrl.types import Transition
from fastpbrl.utils import make_gym_env


class Method(Enum):
    SEQUENTIAL = 0
    VECTORIZED = 1
    PARALLEL = 2


@dataclass
class MeasureConfig:
    use_gpu: bool
    batch_size: int
    num_steps_at_once: int
    population_size: int
    num_iterations: int
    method: Method
    env_name: str
    use_td3: bool
    hidden_layer_sizes: Tuple[int] = (256, 256)

    @staticmethod
    def from_args(args):
        return MeasureConfig(
            use_gpu=args.use_gpu,
            batch_size=args.batch_size,
            num_steps_at_once=args.num_steps_at_once,
            population_size=args.population_size,
            num_iterations=args.num_iterations,
            method=Method[args.method.upper()],
            env_name=args.env_name,
            use_td3=args.use_td3,
        )


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", type=str, help="filepath to output file where to store the results"
    )
    parser.add_argument("--env-name", type=str, default="HalfCheetah-v2")
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--num-steps-at-once", type=int, default=50)
    parser.add_argument(
        "--method",
        type=str,
        choices=[method.name.lower() for method in Method],
        default=Method.SEQUENTIAL.name.lower(),
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use-td3",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    return args


def generate_training_batch(config: MeasureConfig, population_size: int) -> Transition:

    gym_env = make_gym_env(config.env_name)
    obs_dim = gym_env.reset().shape
    action_dim = gym_env.action_space.high.shape
    batch_size = config.batch_size
    num_steps_at_once = config.num_steps_at_once

    transition = Transition(
        observation=np.random.rand(
            population_size, num_steps_at_once, batch_size, *obs_dim
        ).astype(np.float32),
        action=np.random.rand(
            population_size, num_steps_at_once, batch_size, *action_dim
        ).astype(np.float32),
        reward=np.random.rand(population_size, num_steps_at_once, batch_size, 1).astype(
            np.float32
        ),
        done=np.zeros((population_size, num_steps_at_once, batch_size, 1)).astype(
            np.float32
        ),
        next_observation=np.random.rand(
            population_size, num_steps_at_once, batch_size, *obs_dim
        ).astype(np.float32),
    )

    return transition


def default_sac_hyperparams(env_name: str) -> SACHyperParams:
    return SACHyperParams(
        target_entropy=-np.prod(make_gym_env(env_name).action_space.shape, dtype=float)
    )


def measure_runtimes_parallel(
    config: MeasureConfig,
    measure_runtime_func: Callable[
        [
            MeasureConfig,
            Barrier,
            List[Barrier],
            Queue,
        ],
        None,
    ],
) -> Tuple[float, float]:
    set_start_method("spawn")

    # Restrict the amount of memory that can be pre-allocated at startup by jax
    # by each process individually as a function of the population
    # size, otherwise we will run into out-of-memory issues
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "{:.2f}".format(
        0.9 / config.population_size
    )

    population_size = config.population_size
    config = copy.copy(config)
    config.population_size = 1

    all_barriers = [Barrier(parties=i + 1) for i in range(population_size)] + [
        Barrier(parties=population_size)
    ]

    measurements_queue = Queue()
    all_process = [
        Process(
            target=measure_runtime_func,
            kwargs={
                "config": config,
                "barrier_init": all_barriers[process_id],
                "all_barrier_train": all_barriers[process_id + 1 :],
                "measurements_queue": measurements_queue,
            },
            daemon=True,
        )
        for process_id in range(population_size)
    ]

    for process in all_process:
        process.start()

    all_measurements = []
    for _ in range(population_size):
        measurement = measurements_queue.get()
        if measurement is None:
            raise RuntimeError("One measurement failed, aborting...")

        all_measurements.append(measurement)

    avg_runtime = np.mean([avg_runtime for (avg_runtime, _) in all_measurements])
    avg_compilation_time = np.mean(
        [compilation_time for (_, compilation_time) in all_measurements]
    )
    return (avg_runtime, avg_compilation_time)
