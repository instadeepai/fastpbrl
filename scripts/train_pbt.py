import argparse
import datetime
import logging
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import gym
import jax
import jax.numpy as jnp
import numpy as np
import tree
from tensorboardX import SummaryWriter

from fastpbrl import shared_memory as sm
from fastpbrl import utils
from fastpbrl.agents.sac_pbt import SACPBT
from fastpbrl.agents.td3_pbt import TD3PBT
from fastpbrl.evaluate import evaluate
from fastpbrl.non_blocking_iterable import MultiThreadedNonBlockingIterable
from fastpbrl.pbt import PBTExploitStrategy, PBTManager, TruncationPBTExploit
from fastpbrl.replay_buffer import ReplayBufferConfig, ReplayBufferWithSampleRatio
from fastpbrl.step_generator import step_generator
from fastpbrl.types import Transition

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)


@dataclass
class PBTConfig:
    env_name: str = "HalfCheetah-v2"  # Gym environment name
    agent: Union[Type[TD3PBT], Type[SACPBT]] = TD3PBT
    seed: Optional[int] = None  # Seed used for reproducibility
    population_size: int = 80
    num_devices: int = 4  # Number of gpus to use, the population will be split
    # in num_devices groups of agents (one group per device)
    num_actor_processes_per_device: int = 6  # Number of processes (per gpu device) only
    # dedicated to interacting with the environment in order to populate the replay
    # buffers with transition data.
    total_num_env_steps: int = 3_000_000  # Training ends when this total number of
    # environment steps have been carried out by each agent in the population
    use_gpu: bool = True  # If True, the learner looks for and uses a gpu to carry out
    # the update steps
    batch_size: int = 256  # Batch size for computing losses and gradients
    num_update_step_at_once: int = 50  # Frequency, in number of update steps, at
    # which the learners update the actors' with the latest policy parameters
    num_warmup_env_steps: int = 25_000  # The first num_warmup_env_steps actions taken
    # for each agent in the population will be sampled randomly instead of using the
    # policy
    evaluation_frequency: int = 10_000  # Evaluate, for each agent, the policy every
    # evaluation_frequency environment steps carried out jointly by all actors of this
    # agent
    num_episodes_per_evaluation: int = 10  # Number of episodes for each evaluation
    hidden_layer_sizes: Tuple[int] = (256, 256)  # Number and size of the hidden layers
    # of the policy and critic networks
    replay_buffer_config: ReplayBufferConfig = ReplayBufferConfig(
        samples_per_insert=256,  # equal to the batch size so that the target
        # ratio of number of update steps per environment step is one
        insert_error_buffer=200.0,
        min_size_to_sample=25_000,
        max_replay_buffer_size=300_000,
    )
    pbt_update_frequency: int = 100_000  # Run a PBT update across the entire population
    # every time all agents have each carried out pbt_update_frequency number of update
    # steps
    pbt_exploit_strategy: PBTExploitStrategy = TruncationPBTExploit(bottom_frac=0.4)
    transition_queue_batch_size: int = 10  # For efficiency purposes, the actors will
    # send transitions in batches of this size to the learners
    transition_queue_size: int = 30  # At most this number of transitions batches can
    # be waiting in queue to be added to a replay buffer
    num_threads_fetch_batch: int = 6  # Number of threads dedicated to fetching batches
    # from the replay buffer of the agents, concatenating them, and loading them on the
    # GPU
    prefetch_factor: int = 6  # Maximum number of batches to prepare ahead of time for
    # the learners
    num_agents_per_replay_buffer_thread: int = 5  # Number of threads used to populate
    # the replay buffers in the background. Threads are shared between agents for
    # efficiency purposes. Setting this value to 1 will assign one thread for each
    # agent.

    def __post_init__(self):
        self.seed = self.seed if self.seed is not None else random.randint(0, 1_000_000)
        assert self.population_size % self.num_devices == 0


def run_n_actors_sequentially(
    device_id: int,
    process_id: int,
    global_env_step_id: mp.Value,
    shared_memory_policy_params: sm.NestedSharedMemory,
    shared_memory_hyperparams: sm.NestedSharedMemory,
    transition_queue_list: List[mp.Queue],
    information_queue: mp.Queue,
    hyperparams_version: mp.Value,
    params_version: mp.Value,
    config: PBTConfig,
    params_lock: mp.Lock,
):
    spawn_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # hide all GPUs to the actors so that
    # they do not allocate memory to it

    logger = logging.getLogger(f"Actor_device#{device_id}_process#{process_id}")

    # Environment initialization
    eval_environment = utils.make_gym_env(config.env_name)
    max_steps_per_episode = gym.make(config.env_name)._max_episode_steps  # noqa

    seed = config.seed + device_id * config.num_actor_processes_per_device + process_id
    agent = config.agent(
        population_size=config.population_size,
        gym_env_name=config.env_name,
        seed=seed,
        hidden_layer_sizes=config.hidden_layer_sizes,
    )

    with params_lock:
        stacked_policy_params = shared_memory_policy_params.retrieve()
        stacked_hyperparams = shared_memory_hyperparams.retrieve()
        current_params_version = params_version.value
        current_hyperparams_version = hyperparams_version.value

    logger.info("Successfully initialized actor's params")

    policy_params_list = utils.unpack(stacked_policy_params)
    hyperparams_list = utils.unpack(stacked_hyperparams)

    # Initialization of the environment step generators (one per agent)
    env_step_generator_list = [
        step_generator(
            transition_queue=queue,
            policy_params=params,
            hyperparams=hyperparams,
            environment=utils.make_gym_env(config.env_name),
            num_warmup_steps=config.num_warmup_env_steps
            // config.num_actor_processes_per_device,
            transition_queue_batch_size=config.transition_queue_batch_size,
            select_action_function=agent.select_action,
            logger=logger,
            max_steps_per_episode=max_steps_per_episode,
        )
        for (params, hyperparams, queue) in zip(
            policy_params_list, hyperparams_list, transition_queue_list
        )
    ]
    for env_step_generator in env_step_generator_list:
        next(env_step_generator)

    num_episodes_per_evaluation = max(
        1, config.num_episodes_per_evaluation // config.num_actor_processes_per_device
    )

    while True:
        # Get the current global step id
        with global_env_step_id.get_lock():
            current_step_id = global_env_step_id.value
            global_env_step_id.value += 1

        if current_params_version < params_version.value:
            with params_lock:
                stacked_policy_params = shared_memory_policy_params.retrieve()
                current_params_version = params_version.value
            policy_params_list = utils.unpack(stacked_policy_params)

        if current_hyperparams_version < hyperparams_version.value:
            with params_lock:
                stacked_hyperparams = shared_memory_hyperparams.retrieve()
                current_hyperparams_version = hyperparams_version.value
            hyperparams_list = utils.unpack(stacked_hyperparams)

        for seed_id, (_, params, hyperparams) in enumerate(
            zip(env_step_generator_list, policy_params_list, hyperparams_list)
        ):

            step = env_step_generator_list[seed_id].send((params, hyperparams))

            if step.is_episode_done:
                information_queue.put(
                    {
                        "seed_id": seed_id,
                        "device_id": device_id,
                        "return": step.current_return,
                        "length": step.step_id,
                        "step_id": current_step_id,
                        "eval": False,
                        "time": time.time() - spawn_time,
                    }
                )

            if (
                config.evaluation_frequency != 0
                and current_step_id % config.evaluation_frequency == 0
            ):
                return_list = [
                    evaluate(
                        environment=eval_environment,
                        select_action_function=agent.select_action,
                        policy_params=params,
                        hyperparams=hyperparams,
                    )
                    for _ in range(num_episodes_per_evaluation)
                ]
                information_queue.put(
                    {
                        "seed_id": seed_id,
                        "device_id": device_id,
                        "return": np.mean(return_list),
                        "step_id": current_step_id,
                        "eval": True,
                        "time": time.time() - spawn_time,
                    }
                )
                logger.info(f"Evaluation seed #{seed_id} : {np.mean(return_list)}")


def run_learner(
    device_id: int,
    shared_memory_policy_params: sm.NestedSharedMemory,
    shared_memory_hyperparams: sm.NestedSharedMemory,
    transition_queue_list: List[mp.Queue],
    config: PBTConfig,
    params_lock: mp.Lock,
    params_version: mp.Value,
    hyperparams_version: mp.Value,
    barrier: mp.Barrier,
    tensorboard_run_directory: str,
    population_size: int,
    information_queue: mp.Queue,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"  # hide GPUs that will not be
    # used by this learner so that it does not allocate memory on them

    seed = config.seed + device_id + 1
    random.seed(seed)
    np.random.seed(seed)

    spawn_time = time.time()
    writer = SummaryWriter(tensorboard_run_directory, flush_secs=1, write_to_disk=True)
    logger = logging.getLogger(f"Learner_device#{device_id}")

    if config.use_gpu:
        device = jax.devices(backend="gpu")[0]
        logger.info(f"Using GPU {device_id}")
    else:
        device = jax.devices(backend="cpu")[0]

    replay_buffer_list = []
    num_agents_per_replay_buffer_thread = config.num_agents_per_replay_buffer_thread
    num_transition_queue = len(transition_queue_list)
    for i in range(
        (num_transition_queue + num_agents_per_replay_buffer_thread - 1)
        // num_agents_per_replay_buffer_thread
    ):
        replay_buffer_list.append(
            iter(
                ReplayBufferWithSampleRatio(
                    list_transition_queue=transition_queue_list[
                        i
                        * num_agents_per_replay_buffer_thread : (i + 1)
                        * num_agents_per_replay_buffer_thread
                    ],
                    batch_size=config.batch_size * config.num_update_step_at_once,
                    config=config.replay_buffer_config,
                )
            )
        )

    def collate_function(transition_batch_list: List[List[Transition]]) -> Transition:
        all_transition_batch = [
            item for sublist in transition_batch_list for item in sublist
        ]
        shape = (
            population_size,
            config.num_update_step_at_once,
            config.batch_size,
            -1,
        )
        transposed_transitions = utils.numpy_tree_stack(all_transition_batch)
        reshaped_transitions = tree.map_structure(
            lambda x: jnp.reshape(x, shape), transposed_transitions
        )
        return tree.map_structure(
            lambda np_array: jax.device_put(np_array, device).block_until_ready(),
            reshaped_transitions,
        )

    transition_batch_iterator = iter(
        MultiThreadedNonBlockingIterable(
            iterable=zip(*replay_buffer_list),
            collate_fn=collate_function,
            num_threads=config.num_threads_fetch_batch,
            queue_size=config.prefetch_factor,
        )
    )

    # Agent initialization
    agent = config.agent(
        population_size=population_size,
        gym_env_name=config.env_name,
        seed=seed,
        hidden_layer_sizes=config.hidden_layer_sizes,
    )

    # Initialize the neural network weights / optimizer states / hyperparameters
    hyperparams = agent.make_initial_hyperparams()
    shared_memory_hyperparams.update(hyperparams)
    hyperparams = jax.device_put(hyperparams, device)

    training_state = agent.make_initial_training_state()
    shared_memory_policy_params.update(training_state.policy_params)
    training_state = jax.device_put(training_state, device)

    barrier.wait()
    params_lock.release()
    logger.info("Successfully shared learner's params")
    total_num_train_steps = int(
        config.replay_buffer_config.samples_per_insert
        * config.total_num_env_steps
        / config.batch_size
    )

    num_steps_since_pbt_update = 0
    for step_id in range(0, total_num_train_steps, config.num_update_step_at_once):
        start_time_fetch_batch = time.perf_counter()
        transition_batch = next(transition_batch_iterator)

        start_time_update_step = time.perf_counter()
        training_state = agent.update_step(
            training_state=training_state,
            hyperparams=hyperparams,
            transition_batch=transition_batch,
            num_steps=config.num_update_step_at_once,
        )

        start_time_copy_params = time.perf_counter()
        with params_lock:
            shared_memory_policy_params.update(training_state.policy_params)
            params_version.value += 1

        writer.add_scalars(
            main_tag="elapsed_time/learner",
            tag_scalar_dict={
                "fetch_batch": start_time_update_step - start_time_fetch_batch,
                "train_step": (start_time_copy_params - start_time_update_step),
                "copy_params": (time.perf_counter() - start_time_copy_params),
            },
            global_step=step_id,
            walltime=time.time() - spawn_time,
        )

        num_steps_since_pbt_update += config.num_update_step_at_once
        if num_steps_since_pbt_update > config.pbt_update_frequency:
            num_steps_since_pbt_update = 0

            with params_lock:
                # Stopping all the learner at that point so that the PBTManager has
                # received the weights
                barrier.wait()
                # PBTManager is updating hyperparameters / parameters and will hit
                # the barrier again when this operation has completed.
                barrier.wait()

                training_state = training_state._replace(
                    policy_params=jax.device_put(
                        shared_memory_policy_params.retrieve(), device
                    )
                )
                old_hyperparams = jax.tree_map(
                    lambda jnp_array: np.array(jnp_array),
                    jax.device_put(hyperparams, jax.devices("cpu")[0]),
                )
                new_hyperparams = shared_memory_hyperparams.retrieve()
                hyperparams = jax.device_put(new_hyperparams, device)

                params_version.value += 1
                hyperparams_version.value += 1

                updated_agents_indices = []
                for index, (old, new) in enumerate(
                    zip(utils.unpack(old_hyperparams), utils.unpack(new_hyperparams))
                ):
                    if any(
                        jax.tree_flatten(
                            jax.tree_map(lambda a, b: not np.allclose(a, b), old, new)
                        )[0]
                    ):
                        updated_agents_indices.append(index)

                training_state = agent.reset_optimizers(
                    training_state, updated_agents_indices
                )

            logger.info(f"Updated agents during PBT update: {updated_agents_indices}")
            logger.info(f"Hyperparameters before {old_hyperparams=}")
            logger.info(f"Hyperparameters after {hyperparams=}")

    information_queue.put(None)


def main(config: PBTConfig) -> int:
    random.seed(config.seed)
    np.random.seed(config.seed)

    if config.use_gpu and len(jax.devices(backend="gpu")) < config.num_devices:
        raise RuntimeError(
            f"Found {len(jax.devices(backend='gpu'))} accelerators but "
            f"{config.num_devices} were specified in the configuration."
        )

    # Set up the run name as well as the writer directories
    tensorboard_run_name = (
        f'pbt_{config.env_name}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )
    tensorboard_run_directory = (
        Path(__file__).parent.parent.joinpath("logs").joinpath(tensorboard_run_name)
    )
    print(f"tensorboard --logdir {str(tensorboard_run_directory)} --port={str(6006)}")

    # Initialized the global shared_state
    information_queue = mp.Queue()
    barrier = mp.Barrier(config.num_devices + 1)

    # Initialize networks in order to create the shared memory space
    population_size_per_device = config.population_size // config.num_devices
    agent = config.agent(
        population_size=population_size_per_device,
        gym_env_name=config.env_name,
        seed=0,
        hidden_layer_sizes=config.hidden_layer_sizes,
    )

    actors_process_list: List[mp.Process] = []
    learners_process_list: List[mp.Process] = []
    device_id_to_shared_params: Dict[int, sm.NestedSharedMemory] = {}
    device_id_to_shared_hyperparams: Dict[int, sm.NestedSharedMemory] = {}
    all_mp_variables = []

    # Allocate shared memory for storing policy params
    with SharedMemoryManager() as shared_memory_manager:
        for device_id in range(config.num_devices):
            # Initialize the objects that enable cross-communication between the
            # processes

            transition_queue_list = [
                mp.Queue(maxsize=config.transition_queue_size)
                for _ in range(population_size_per_device)
            ]  # queues for sending transition data from the actors to the learner
            parameters_version = mp.Value(
                "i", 0
            )  # Shared variable incremented every time a
            # new version of the policy parameter is made available to the actors
            hyperparams_version = mp.Value(
                "i", 0
            )  # Shared variable incremented every time a
            # new version of the hyperparameters is made available to the actors
            global_env_step_id = mp.Value(
                "i", 0
            )  # Shared variable counting the total number
            # of environment steps carried out so far joitnly by all the actors
            params_lock = mp.Lock()  # Lock for reading the policy parameters and
            # hyperparameters from shared memory
            params_lock.acquire()
            shared_memory_policy_params = sm.NestedSharedMemory(
                nested_array=agent.make_initial_training_state().policy_params,
                shared_memory_manager=shared_memory_manager,
            )
            device_id_to_shared_params[device_id] = shared_memory_policy_params
            shared_memory_hyperparams = sm.NestedSharedMemory(
                nested_array=agent.make_initial_hyperparams(),
                shared_memory_manager=shared_memory_manager,
            )
            device_id_to_shared_hyperparams[device_id] = shared_memory_hyperparams

            all_mp_variables.extend(
                [
                    transition_queue_list,
                    parameters_version,
                    hyperparams_version,
                    global_env_step_id,
                    params_lock,
                    shared_memory_policy_params,
                    shared_memory_hyperparams,
                ]
            )

            actors_process_list.extend(
                [
                    mp.Process(
                        target=run_n_actors_sequentially,
                        kwargs={
                            "device_id": device_id,
                            "process_id": process_id,
                            "global_env_step_id": global_env_step_id,
                            "shared_memory_policy_params": shared_memory_policy_params,
                            "shared_memory_hyperparams": shared_memory_hyperparams,
                            "transition_queue_list": transition_queue_list,
                            "information_queue": information_queue,
                            "config": config,
                            "hyperparams_version": hyperparams_version,
                            "params_version": parameters_version,
                            "params_lock": params_lock,
                        },
                        daemon=True,
                    )
                    for process_id in range(config.num_actor_processes_per_device)
                ]
            )

            learners_process_list.append(
                mp.Process(
                    target=run_learner,
                    kwargs={
                        "device_id": device_id,
                        "shared_memory_policy_params": shared_memory_policy_params,
                        "shared_memory_hyperparams": shared_memory_hyperparams,
                        "transition_queue_list": transition_queue_list,
                        "information_queue": information_queue,
                        "barrier": barrier,
                        "config": config,
                        "params_version": parameters_version,
                        "hyperparams_version": hyperparams_version,
                        "params_lock": params_lock,
                        "tensorboard_run_directory": str(
                            tensorboard_run_directory / f"learner_d{device_id}"
                        ),
                        "population_size": population_size_per_device,
                    },
                    daemon=True,
                )
            )

        # Launch PBT Manager in a process to be able to terminate it when learner
        # finished
        pbt_manager_process = mp.Process(
            target=PBTManager(
                config=config,
                information_queue=information_queue,
                device_id_to_shared_params=device_id_to_shared_params,
                pbt_exploit_strategy=config.pbt_exploit_strategy,
                barrier=barrier,
                run_directory_path=tensorboard_run_directory,
                device_id_to_shared_hyperparams=device_id_to_shared_hyperparams,
            ).run,
            daemon=True,
        )
        pbt_manager_process.start()

        for process in actors_process_list + learners_process_list:
            process.start()

        for process in learners_process_list:
            process.join()

        for process in actors_process_list + [pbt_manager_process]:
            process.terminate()

    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-sac",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    config = PBTConfig(agent=SACPBT if args.use_sac else TD3PBT)
    main(config)
