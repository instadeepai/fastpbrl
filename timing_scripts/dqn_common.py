import argparse
import copy
import os
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Barrier, Process, Queue, set_start_method
from typing import Callable, List, Tuple

import numpy as np


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
    num_frame_stack: int = 4
    num_actions: int = 14
    image_shape: Tuple[int] = (84, 84)
    hidden_layer_sizes: Tuple[int] = (256,)

    @staticmethod
    def from_args(args):
        return MeasureConfig(
            use_gpu=args.use_gpu,
            batch_size=args.batch_size,
            num_steps_at_once=args.num_steps_at_once,
            population_size=args.population_size,
            num_iterations=args.num_iterations,
            method=Method[args.method.upper()],
        )


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", type=str, help="filepath to output file where to store the results"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps-at-once", type=int, default=50)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument(
        "--method",
        type=str,
        choices=[method.name.lower() for method in Method],
        default=Method.SEQUENTIAL.name.lower(),
    )
    args = parser.parse_args()
    return args


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

    # Restrict the amount of memory that can be pre-allocated at starup by jax
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

    for process in all_process:
        process.join()

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
