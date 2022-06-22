import argparse
import datetime
import logging
import os
import random
import time
from dataclasses import dataclass
from multiprocessing import Lock, Process, Queue, Value, set_start_method
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import List, Optional, Tuple, Union

import gym
import jax
import numpy as np
import tree
from tensorboardX import SummaryWriter

from fastpbrl import shared_memory as sm
from fastpbrl.agents.sac import SAC
from fastpbrl.agents.td3 import TD3
from fastpbrl.evaluate import evaluate
from fastpbrl.non_blocking_iterable import MultiThreadedNonBlockingIterable
from fastpbrl.replay_buffer import ReplayBufferConfig, ReplayBufferWithSampleRatio
from fastpbrl.sac.core import SACHyperParams
from fastpbrl.step_generator import step_generator
from fastpbrl.td3.core import TD3HyperParams
from fastpbrl.types import Transition
from fastpbrl.utils import make_gym_env, write_config

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)


@dataclass
class DistributedConfig:
    env_name: str = "HalfCheetah-v2"  # Gym environment name
    hyperparams: Union[TD3HyperParams, SACHyperParams] = TD3HyperParams()
    seed: Optional[int] = None  # Seed used for reproducibility
    # Total number of environment steps performed by the agent
    total_num_env_steps: int = 1_000_000  # Training ends when this total number
    # of environment steps have been carried out jointly by all of the actors
    use_gpu: bool = True  # If True, the learner looks for and uses a gpu to carry
    # out the update steps
    batch_size: int = 256  # Batch size for computing losses and gradients
    num_update_step_at_once: int = 50  # Frequency, in number of update steps, at
    # which the learner updates the actors' with the latest policy parameters
    num_actors: int = 4  # Number of processes only dedicated to interacting with the
    # environment in order to populate the replay buffer with transition data.
    num_warmup_env_steps: int = 25_000  # The first num_warmup_env_steps actions taken
    # by the actors will be sampled randomly instead of using the policy
    evaluation_frequency: int = 5_000  # Evaluate the policy every evaluation_frequency
    # environment steps carried out jointly by all actors
    num_episodes_per_evaluation: int = 10  # Number of episodes for each evaluation
    hidden_layer_sizes: Tuple[int] = (256, 256)  # Number and size of the hidden layers
    # of the policy and critic networks
    replay_buffer_config: ReplayBufferConfig = ReplayBufferConfig(
        samples_per_insert=256.0,  # equal to the batch size so that the target
        # ratio of number of update steps per environment step is one
        insert_error_buffer=100.0,
        min_size_to_sample=25_000,
        max_replay_buffer_size=1_000_000,
    )
    transition_queue_batch_size: int = 10  # For efficiency purposes, the actors will
    # send transitions in batches of this size to the learner
    transition_queue_size: int = 20  # At most this number of transitions batches can
    # be waiting in queue to be added to the replay buffer

    def __post_init__(self):
        self.seed = self.seed if self.seed is not None else random.randint(0, 1_000_000)
        if isinstance(self.hyperparams, SACHyperParams):
            self.agent = SAC
        elif isinstance(self.hyperparams, TD3HyperParams):
            self.agent = TD3
        else:
            raise NotImplementedError('Only "TD3" and "SAC" are implemented')


def run_actor(
    actor_id: int,
    shared_memory: sm.NestedSharedMemory,
    transition_queue: Queue,
    config: DistributedConfig,
    policy_params_lock: Lock,
    parameters_version: Value,
    global_env_step_id: Value,
    tensorboard_run_directory: str,
):
    spawn_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # hide all GPUs to the actors so that
    # they do not allocate memory to it

    # Logging initialization
    tensorboard_writer = SummaryWriter(
        tensorboard_run_directory, flush_secs=1, write_to_disk=True
    )
    logger = logging.getLogger(f"Actor#{actor_id}")

    # Environments initialization
    environment = make_gym_env(config.env_name)
    eval_environment = make_gym_env(config.env_name)
    max_steps_per_episode = gym.make(config.env_name)._max_episode_steps  # noqa

    # Actor initialization
    agent = config.agent(
        gym_env_name=config.env_name,
        seed=config.seed + actor_id,
        hidden_layer_sizes=config.hidden_layer_sizes,
    )

    # Fetch initial value of policy parameters
    with policy_params_lock:
        policy_params = shared_memory.retrieve()
        current_params_version = parameters_version.value

    # Initialization of the environment step generator
    env_step_generator = step_generator(
        transition_queue=transition_queue,
        policy_params=policy_params,
        hyperparams=config.hyperparams,
        environment=environment,
        num_warmup_steps=config.num_warmup_env_steps // config.num_actors,
        max_steps_per_episode=max_steps_per_episode,
        transition_queue_batch_size=config.transition_queue_batch_size,
        select_action_function=agent.select_action,
        logger=logger,
    )
    step = next(env_step_generator)

    num_episodes_per_evaluation = (
        config.num_episodes_per_evaluation // config.num_actors
    )
    global_env_step_id_at_last_eval = 0
    while True:
        # Update the global step id
        with global_env_step_id.get_lock():
            global_env_step_id.value += 1

        # Fetch the policy parameters from the shared memory with the learner
        # but only if a new version of the parameters is available
        if current_params_version < parameters_version.value:
            with policy_params_lock:
                policy_params = shared_memory.retrieve()
                current_params_version = parameters_version.value

        # Step the environment using the latest policy parameters available
        step = env_step_generator.send((policy_params, config.hyperparams))
        logger.debug(f"Ongoing episode #{step.episode_id}, step: {step.step_id}")

        # Evaluate the current policy parameters if necessary
        if (
            global_env_step_id_at_last_eval == 0
            or (global_env_step_id.value - global_env_step_id_at_last_eval)
            >= config.evaluation_frequency
        ):
            global_env_step_id_at_last_eval = global_env_step_id.value
            mean_return = np.mean(
                [
                    evaluate(
                        environment=eval_environment,
                        select_action_function=agent.select_action,
                        policy_params=policy_params,
                        hyperparams=config.hyperparams,
                    )
                    for _ in range(num_episodes_per_evaluation)
                ]
            )

            tensorboard_writer.add_scalar(
                tag="evaluation/episode_return",
                scalar_value=mean_return,
                global_step=global_env_step_id.value,
                walltime=time.time() - spawn_time,
            )


def run_learner(
    shared_memory: sm.NestedSharedMemory,
    global_env_step_id: Value,
    transition_queue: Queue,
    config: DistributedConfig,
    policy_params_lock: Lock,
    parameters_version: Value,
    tensorboard_run_directory: str,
):
    spawn_time = time.time()

    # Logging initialization
    logger = logging.getLogger("Learner")
    tensorboard_writer = SummaryWriter(
        tensorboard_run_directory, flush_secs=1, write_to_disk=True
    )
    write_config(writer=tensorboard_writer, config=config)

    # Determine jax device to use (gpu or cpu)
    device = jax.devices(backend="gpu" if config.use_gpu else "cpu")[0]

    # Replay buffer initialization
    replay_buffer = ReplayBufferWithSampleRatio(
        list_transition_queue=[transition_queue],
        batch_size=config.batch_size * config.num_update_step_at_once,
        config=config.replay_buffer_config,
    )

    # Agent initialization
    agent = config.agent(
        gym_env_name=config.env_name,
        seed=config.seed,
        hidden_layer_sizes=config.hidden_layer_sizes,
    )

    # Initialize the neural network weights / optimizer states
    training_state = agent.make_initial_training_state()
    training_state = jax.device_put(training_state, device)

    def collate_function(list_transition_batch: List[Transition]) -> Transition:
        shape = (config.num_update_step_at_once, config.batch_size, -1)
        return tree.map_structure(
            lambda np_array: jax.device_put(
                np_array.reshape(shape), device
            ).block_until_ready(),
            list_transition_batch[0],
        )

    transition_batch_iterator = iter(
        MultiThreadedNonBlockingIterable(
            iterable=replay_buffer,
            collate_fn=collate_function,
            num_threads=1,
        )
    )

    while global_env_step_id.value < config.total_num_env_steps:
        start_time_fetch_batch = time.perf_counter()
        transition_batch = next(transition_batch_iterator)

        start_time_update_step = time.perf_counter()
        training_state = agent.update_step(
            training_state,
            hyperparams=config.hyperparams,
            transition_batch=transition_batch,
            num_steps=config.num_update_step_at_once,
        )

        # Copy the new policy weights to the shared memory for the actors
        start_time_copy_params = time.perf_counter()
        with policy_params_lock:
            shared_memory.update(training_state.policy_params)
            parameters_version.value += 1

        tensorboard_writer.add_scalars(
            main_tag="elapsed_time/learner",
            tag_scalar_dict={
                "fetch_batch": start_time_update_step - start_time_fetch_batch,
                "train_step": (start_time_copy_params - start_time_update_step),
                "copy_params": (time.perf_counter() - start_time_copy_params),
            },
            global_step=global_env_step_id.value,
            walltime=time.time() - spawn_time,
        )

    logger.warning(
        "Training has ended, we have reached the total number of environment steps "
        f"{config.total_num_env_steps}"
    )


def main(config: DistributedConfig) -> int:
    # Set-up the run name for tensorboard as well as the tensorboard logs directories
    tensorboard_run_name = (
        f'distributed_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    tensorboard_run_directory = (
        Path(__file__).parent.parent.joinpath("logs").joinpath(tensorboard_run_name)
    )
    print(f"tensorboard --logdir {str(tensorboard_run_directory)} --port={str(6006)}")

    # Initialize the objects that enable cross-communication between the processes
    transition_queue = Queue(maxsize=config.transition_queue_size)  # queue for
    # sending transition data from the actors to the learner
    global_env_step_id = Value("i", 0)  # Shared variable counting the total number
    # of environment steps carried out so far joitnly by all the actors
    parameters_version = Value("i", 0)  # Shared variable incremented every time a
    # new version of the policy parameter is made available to the actors
    policy_params_lock = Lock()  # Lock for reading the policy parameters from
    # shared memory

    with SharedMemoryManager() as shared_memory_manager:
        # Allocate shared memory for storing the policy parameters and
        # sharing them with the actors
        agent = config.agent(
            gym_env_name=config.env_name,
            seed=config.seed,
            hidden_layer_sizes=config.hidden_layer_sizes,
        )
        shared_memory = sm.NestedSharedMemory(
            nested_array=agent.make_initial_training_state().policy_params,
            shared_memory_manager=shared_memory_manager,
        )

        actor_process_list = [
            Process(
                target=run_actor,
                kwargs={
                    "actor_id": actor_id,
                    "shared_memory": shared_memory,
                    "global_env_step_id": global_env_step_id,
                    "transition_queue": transition_queue,
                    "config": config,
                    "parameters_version": parameters_version,
                    "policy_params_lock": policy_params_lock,
                    "tensorboard_run_directory": str(
                        tensorboard_run_directory / f"actor_{actor_id}"
                    ),
                },
                daemon=True,
            )
            for actor_id in range(config.num_actors)
        ]

        for actor_process in actor_process_list:
            actor_process.start()

        run_learner(
            shared_memory=shared_memory,
            global_env_step_id=global_env_step_id,
            transition_queue=transition_queue,
            config=config,
            parameters_version=parameters_version,
            policy_params_lock=policy_params_lock,
            tensorboard_run_directory=str(tensorboard_run_directory / "learner"),
        )

        for actor_process in actor_process_list:
            actor_process.terminate()

        return 0


if __name__ == "__main__":
    set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-sac",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    config = DistributedConfig(
        hyperparams=SACHyperParams() if args.use_sac else TD3HyperParams()
    )
    main(config)
