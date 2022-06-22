import multiprocessing
import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from multiprocessing import Barrier, Queue
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from tensorboardX import SummaryWriter

import fastpbrl.shared_memory as sm
from fastpbrl import utils
from fastpbrl.types import HyperParams
from fastpbrl.utils import tree_index_select, tree_update

# The seed key is composed of the device_id and the seed_id
SeedKey = Tuple[int, int]


class PBTExploitStrategy(ABC):
    @abstractmethod
    def exploit(
        self, seed_key_to_reward: Dict[SeedKey, float]
    ) -> Dict[SeedKey, SeedKey]:
        """Exploit will use the dictionary of agent to reward in order to create an
        update dictionary that will map the worst (target) agent's key to the best
        (source) agent's key
        """


@dataclass
class TruncationPBTExploit(PBTExploitStrategy):
    bottom_frac: float = 0.3

    def exploit(
        self, seed_key_to_return: Dict[SeedKey, float]
    ) -> Dict[SeedKey, SeedKey]:
        population_size = len(seed_key_to_return.keys())
        assert seed_key_to_return
        assert 0 < int(self.bottom_frac * population_size) <= population_size / 2, (
            f"0 < {int(self.bottom_frac * population_size)} <= {population_size / 2} \n"
            f" with population_size={population_size} and "
            f"bottom_frac={self.bottom_frac}"
        )

        threshold_index = int(self.bottom_frac * population_size)
        threshold_value = sorted(seed_key_to_return.values())[threshold_index]

        best_seed_list = []
        worst_seed_list = []
        for seed_key, return_ in seed_key_to_return.items():
            if return_ < threshold_value:
                worst_seed_list.append(seed_key)
            else:
                best_seed_list.append(seed_key)

        swap_dict: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for worst_seed in worst_seed_list:
            swap_dict[worst_seed] = random.choice(best_seed_list)

        return swap_dict


@dataclass
class PBTManager:
    config: Any
    pbt_exploit_strategy: PBTExploitStrategy
    information_queue: Queue
    barrier: Barrier
    device_id_to_shared_params: Dict[int, sm.NestedSharedMemory]
    device_id_to_shared_hyperparams: Dict[int, sm.NestedSharedMemory]
    run_directory_path: Path

    def __post_init__(self):
        self._logger = getLogger("PBTManager")
        self._thread: Optional[threading.Thread] = None
        self._lock: Optional[multiprocessing.Lock] = None
        self._device_id_seed_id_to_return: Dict[SeedKey, float] = {}
        self._device_id_seed_id_to_writer: Dict[SeedKey, SummaryWriter] = {}
        self._hyperparams_sampler: Optional[Callable[[], HyperParams]] = None

    def _initialization(self):
        """The initialization is done after that the PBTManager is spawned, otherwise
        it is impossible to pickle the object (writers, ...)
        """
        self._hyperparams_sampler = self.config.agent(
            population_size=self.config.population_size,
            gym_env_name=self.config.env_name,
            seed=self.config.seed,
            hidden_layer_sizes=self.config.hidden_layer_sizes,
        ).get_hyperparams_sampler_function()

        # Wait that all the learner are initialized
        self.barrier.wait()

        # Initialization of the writers
        self._device_id_seed_id_to_writer = {}
        for device_id in range(self.config.num_devices):
            for seed_id in range(self.config.population_size):
                path = str(self.run_directory_path / f"actor_d{device_id}_s{seed_id}")
                writer = SummaryWriter(path, flush_secs=1, write_to_disk=True)
                self._device_id_seed_id_to_writer[(device_id, seed_id)] = writer

        # Initialization of the thread and the lock for retrieving the agents' returns
        self._lock = threading.Lock()
        self._device_id_seed_id_to_return: Dict[SeedKey, float] = {}
        self._thread = threading.Thread(
            target=self._background_thread, args=(), daemon=True
        )
        self._thread.start()

    def run(self):
        # Initialization
        self._initialization()

        # PBT Iteration
        while True:
            # Wait that all the agents signal that they need updated hyperparams
            self.barrier.wait()

            # Create the swap dictionary
            with self._lock:
                self._logger.info(f"{self._device_id_seed_id_to_return=}")
                swap_dict = self.pbt_exploit_strategy.exploit(
                    self._device_id_seed_id_to_return
                )
            self._logger.info(f"{swap_dict=}")

            # Loop through the swap dict to update the params and hyperparams
            for target_key, source_key in swap_dict.items():
                self._update_seed(source_key=source_key, target_key=target_key)

            self.barrier.wait()

    def _update_seed(self, source_key: SeedKey, target_key: SeedKey):
        (target_device_id, target_seed_id) = target_key
        (source_device_id, source_seed_id) = source_key

        params = tree_index_select(
            source=self.device_id_to_shared_params[source_device_id].retrieve(),
            target=self.device_id_to_shared_params[
                target_device_id
            ].nested_shared_numpy_array,
            source_index=source_seed_id,
            target_index=target_seed_id,
        )
        self.device_id_to_shared_params[target_device_id].update(
            source_nested_array=params
        )

        hyperparams = tree_update(
            source=self._hyperparams_sampler(),
            target=self.device_id_to_shared_hyperparams[
                target_device_id
            ].nested_shared_numpy_array,
            target_index=target_seed_id,
        )
        self.device_id_to_shared_hyperparams[target_device_id].update(
            source_nested_array=hyperparams
        )

    def _background_thread(self):
        """Thread to fetch the return values of the agents, populate the return
        dictionary and log the results"""
        stop_loop_counter = 0
        while stop_loop_counter < self.config.num_devices:
            info = self.information_queue.get()

            # The learner will send None when they finish to learn
            if info is None:
                stop_loop_counter += 1
                continue

            # Update the PBT dictionary
            elif info["eval"]:
                with self._lock:
                    self._device_id_seed_id_to_return[
                        (info["device_id"], info["seed_id"])
                    ] = info["return"]

            # Logging
            tag = "eval" if info["eval"] else "actor"
            utils.write_dict(
                writer=self._device_id_seed_id_to_writer[
                    (info["device_id"], info["seed_id"])
                ],
                tag_to_scalar_value={
                    f"{tag}/episode_return": info["return"],
                },
                global_step=info["step_id"],
                walltime=info["time"],
            )
        return
