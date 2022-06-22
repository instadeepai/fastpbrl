import datetime
import logging
import os
import random
import time
from dataclasses import dataclass
from multiprocessing import Event, Process, Queue, Value, set_start_method
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import gym
import jax
import numpy as np
import tree
from tensorboardX import SummaryWriter

from fastpbrl import shared_memory as sm
from fastpbrl.agents.td3_cemrl import TD3CEMRL
from fastpbrl.cem import CrossEntropyMethod, CrossEntropyParameters
from fastpbrl.evaluate import evaluate
from fastpbrl.non_blocking_iterable import MultiThreadedNonBlockingIterable
from fastpbrl.replay_buffer import ReplayBufferConfig, ReplayBufferWithSampleRatio
from fastpbrl.step_generator import step_generator
from fastpbrl.td3.core import TD3HyperParams
from fastpbrl.types import Transition
from fastpbrl.utils import jax_tree_stack, make_gym_env, write_config

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)


@dataclass
class CEMRLTD3Config:
    env_name: str = "Hopper-v2"  # Gym environment name
    seed: Optional[int] = None  # Seed used for reproducibility
    # Total number of environment steps performed by the agent
    total_num_env_steps: int = 1_000_000  # Training ends when this total number
    # of environment steps have been carried out jointly by all of the actors
    use_gpu: bool = True  # If True, the learner looks for and uses a gpu to carry
    # out the update steps
    population_size: int = 10
    num_policies_to_train: Optional[int] = None  # Number of policies that are trained
    # using TD3. If None this will default to population_size / 2
    cem_parameters: CrossEntropyParameters = CrossEntropyParameters(
        num_elites=5,
    )  # Parameters to use for CEM
    num_episodes_for_fitness: int = 1  # Number of episodes used to evaluate a newly
    # available policy at every CEM-RL iteration
    num_episodes_for_eval: int = 10  # Number of episodes used to evaluate the average
    # policy at every CEM-RL iteration
    batch_size: int = 256  # Batch size for computing losses and gradients
    num_warmup_env_steps: int = 1_000  # TD3 Training starts when this total number of
    # environment steps have been carried out jointly by all of the actors
    num_update_step_at_once: int = 50  # Update steps are performed by batches of
    # num_update_step_at_once without retrieving the parameters in between. This
    # is for efficiency purposes to avoid back-and-forth communication between the
    # GPU and the cpu
    td3_hyperparams: TD3HyperParams = TD3HyperParams(delay=1)
    hidden_layer_sizes: Tuple[int] = (256, 256)  # Number and size of the hidden layers
    # of the policy and critic networks
    max_replay_buffer_size: int = 1_000_000  # Maximum number of transitions stored in
    # the replay buffer, transitions are deleted in a FIFO fashion afterwards
    transition_queue_batch_size: int = 10  # For efficiency purposes, the actors will
    # send transitions in batches of this size to the learner

    def __post_init__(self):
        self.seed = self.seed if self.seed is not None else random.randint(0, 1_000_000)
        self.num_policies_to_train = (
            self.num_policies_to_train
            if self.num_policies_to_train is not None
            else (self.population_size + 1) // 2
        )
        assert self.num_policies_to_train <= self.population_size


class EvaluationInfo(NamedTuple):
    """
    Feedback sent from an actor to the learner at the end of a CEM-RL iteration
    to report on the performance of the evaluated policy.
    """

    actor_id: int
    num_env_steps: int  # Total number of environment steps carried out by this
    # actor during the iteration
    mean_episode_return: float  # Average episode return measured over
    # num_episodes_for_fitness episodes


def run_actor(
    actor_id: int,
    shared_memory: sm.NestedSharedMemory,
    transition_queue: Queue,
    evaluations_queue: Queue,
    config: CEMRLTD3Config,
    global_env_step_id: Value,
    update_params_event: Event,
    end_of_eval_event: Event,
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
    agent = TD3CEMRL(
        gym_env_name=config.env_name,
        population_size=1,
        seed=config.seed + actor_id,
        hidden_layer_sizes=config.hidden_layer_sizes,
    )

    # Fetch initial value of policy parameters
    update_params_event.wait()
    policy_params = shared_memory.retrieve_with_index(actor_id)
    mean_policy_params = shared_memory.retrieve_with_index(config.population_size)

    # Initialization of the environment step generator
    env_step_generator = step_generator(
        policy_params=policy_params,
        hyperparams=config.td3_hyperparams,
        environment=environment,
        select_action_function=agent.select_action,
        max_steps_per_episode=max_steps_per_episode,
        logger=logger,
        transition_queue_batch_size=config.transition_queue_batch_size,
        transition_queue=transition_queue,
        never_explore=True,
    )
    step = next(env_step_generator)

    # Determine the number of episodes used to evaluate the mean policy at every
    # iteration for this actor
    num_episodes_for_eval = config.num_episodes_for_eval // config.population_size
    if config.num_episodes_for_eval % config.population_size != 0:
        num_episodes_for_eval += 1

    iter_id = 0
    while True:
        iter_id += 1

        logger.info(f"Start of fitness evaluation for iteration #{iter_id}")
        list_episode_return: List[float] = []
        num_env_steps = 0
        for _ in range(config.num_episodes_for_fitness):
            while True:
                # Update the global step id
                with global_env_step_id.get_lock():
                    global_env_step_id.value += 1

                # Step the environment using the latest policy parameters
                step = env_step_generator.send((policy_params, config.td3_hyperparams))
                num_env_steps += 1
                if step.is_episode_done:
                    list_episode_return.append(step.current_return)
                    break

        evaluations_queue.put(
            EvaluationInfo(
                actor_id=actor_id,
                mean_episode_return=np.mean(list_episode_return),
                num_env_steps=num_env_steps,
            )
        )
        logger.info(f"Done evaluating fitness for iteration #{iter_id}")

        end_of_eval_event.wait()  # Wait for the other actors to finish evaluating
        # their policies

        logger.info(f"Start evaluating the mean policy for iteration #{iter_id}")
        mean_return = np.mean(
            [
                evaluate(
                    environment=eval_environment,
                    select_action_function=agent.select_action,
                    policy_params=mean_policy_params,
                    hyperparams=config.td3_hyperparams,
                )
                for _ in range(num_episodes_for_eval)
            ]
        )
        logger.info(f"Done evaluating the mean policy for iteration #{iter_id}")

        tensorboard_writer.add_scalar(
            tag="evaluate/return",
            scalar_value=mean_return,
            global_step=global_env_step_id.value,
            walltime=time.time() - spawn_time,
        )

        logger.info("Waiting for new policy parameters to evaluate")
        update_params_event.wait()
        policy_params = shared_memory.retrieve_with_index(actor_id)
        mean_policy_params = shared_memory.retrieve_with_index(config.population_size)


def run_learner(
    shared_memory: sm.NestedSharedMemory,
    global_env_step_id: Value,
    transition_queue: Queue,
    evaluations_queue: Queue,
    config: CEMRLTD3Config,
    update_params_event: Event,
    end_of_eval_event: Event,
    tensorboard_run_directory: str,
):
    spawn_time = time.time()

    # Logging initialization
    logger = logging.getLogger("Learner")
    tensorboard_writer = SummaryWriter(
        tensorboard_run_directory, flush_secs=1, write_to_disk=True
    )
    write_config(tensorboard_writer, config)

    # Determine jax device to use (gpu or cpu)
    device = jax.devices(backend="gpu" if config.use_gpu else "cpu")[0]

    # Replay buffer initialization
    replay_buffer = ReplayBufferWithSampleRatio(
        list_transition_queue=[transition_queue],
        batch_size=config.batch_size
        * config.num_update_step_at_once
        * config.num_policies_to_train,
        config=ReplayBufferConfig(
            min_size_to_sample=config.num_warmup_env_steps,
            max_replay_buffer_size=config.max_replay_buffer_size,
        ),
    )

    # Agent initialization
    agent = TD3CEMRL(
        gym_env_name=config.env_name,
        population_size=config.num_policies_to_train,
        seed=config.seed,
        hidden_layer_sizes=config.hidden_layer_sizes,
    )

    # Initialize the neural network weights and optimizer states
    training_state = agent.make_initial_training_state()

    # Initialize the CEM state using the first policy parameters
    first_policy_params = jax.tree_map(
        lambda jax_array: jax_array[0], training_state.policy_params
    )
    cross_entropy_method = CrossEntropyMethod(
        cem_params=config.cem_parameters, initial_params=first_policy_params
    )

    def collate_function(list_transition_batch: List[Transition]) -> Transition:
        shape = (
            config.num_update_step_at_once,
            config.num_policies_to_train,
            config.batch_size,
            -1,
        )
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

    last_iter_num_env_steps = 0  # Number of environment steps carried joinly
    # by all actors for the previous CEM-RL iteration
    iter_id = 0
    while global_env_step_id.value < config.total_num_env_steps:
        logger.info(f"Beginning of CEM-RL iteration #{iter_id}")
        iter_id += 1

        # Sample new policy parameters using CEM
        new_list_policy_params = cross_entropy_method.sample(config.population_size)

        # Stack the new policy parameters along the first axis and copy it to
        # shared memory so that the actors can fetch their policy.
        # Also include the mean value so that the actors can evaluate it.
        policy_params = jax_tree_stack(
            new_list_policy_params + [cross_entropy_method.get_mean()]
        )
        shared_memory.update(policy_params)

        if global_env_step_id.value > config.num_warmup_env_steps:
            # Override the policy parameters stored in training_state to
            # the values sampled by CEM.
            training_state = training_state._replace(
                policy_params=jax.tree_map(
                    lambda x: x[: config.num_policies_to_train], policy_params
                ),
            )
            # Reset the policy optimizer states
            training_state = agent.reset_optimizers(training_state)
            training_state = jax.device_put(training_state, device)

            # Carry out last_iter_num_env_steps update steps
            for _ in range(last_iter_num_env_steps // config.num_update_step_at_once):
                transition_batch = next(transition_batch_iterator)

                start_time_update_step = time.perf_counter()
                training_state = agent.update_step(
                    training_state,
                    hyperparams=config.td3_hyperparams,
                    transition_batch=transition_batch,
                    num_steps=config.num_update_step_at_once,
                )

                tensorboard_writer.add_scalar(
                    tag="elapsed_time/learner/train_step",
                    scalar_value=(time.perf_counter() - start_time_update_step),
                    global_step=global_env_step_id.value,
                    walltime=time.time() - spawn_time,
                )

            # Copy the updated policy weights to shared memroy for the actors
            shared_memory.update(training_state.policy_params, is_subarray=True)

        # Signal to the actors that new policy parameters are available and
        # should be evaluated
        end_of_eval_event.clear()
        update_params_event.set()

        logger.info("Waiting for actors to evaluate the policies")
        last_iter_num_env_steps = 0
        for _ in range(config.population_size):
            evaluation_info = evaluations_queue.get()
            single_policy_params = shared_memory.retrieve_with_index(
                evaluation_info.actor_id
            )
            cross_entropy_method.add(
                single_policy_params, fitness=evaluation_info.mean_episode_return
            )
            last_iter_num_env_steps += evaluation_info.num_env_steps
        logger.info("Received all actors' evaluations")

        end_of_eval_event.set()
        update_params_event.clear()


def main(config: CEMRLTD3Config):
    # Set-up the run name for tensorboard as well as the tensorboard logs directories
    tensorboard_run_name = (
        f'td3cemrl_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    tensorboard_run_directory = (
        Path(__file__).parent.parent.joinpath("logs").joinpath(tensorboard_run_name)
    )
    print(f"tensorboard --logdir {str(tensorboard_run_directory)} --port={str(6006)}")

    # Initialize the objects that enable cross-communication between the processes
    transition_queue = Queue()  # queue for sending transition data from the actors to
    # the learner
    evaluations_queue = Queue()  # queue for sending EvaluationInfo data from the actors
    # to the learner
    global_env_step_id = Value("i", 0)  # Shared variable counting the total number
    # of environment steps carried out so far joitnly by all the actors
    update_params_event = Event()  # event to signal to the actors that new policies
    # are available to be evaluated
    end_of_eval_event = Event()  # event to signal to the actors that all the other
    # actors have finished evaluating their policy parameters

    with SharedMemoryManager() as shared_memory_manager:
        # Allocate shared memory for storing the policy parameters
        # for the entire population (+ 1 to also store the average policy parameters)
        # and to share them with the actors
        agent = TD3CEMRL(
            gym_env_name=config.env_name,
            population_size=config.population_size + 1,
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
                    "transition_queue": transition_queue,
                    "evaluations_queue": evaluations_queue,
                    "config": config,
                    "global_env_step_id": global_env_step_id,
                    "update_params_event": update_params_event,
                    "end_of_eval_event": end_of_eval_event,
                    "tensorboard_run_directory": str(
                        tensorboard_run_directory / f"actor_{actor_id}"
                    ),
                },
                daemon=True,
            )
            for actor_id in range(config.population_size)
        ]

        for actor_process in actor_process_list:
            actor_process.start()

        run_learner(
            shared_memory=shared_memory,
            global_env_step_id=global_env_step_id,
            transition_queue=transition_queue,
            evaluations_queue=evaluations_queue,
            config=config,
            update_params_event=update_params_event,
            end_of_eval_event=end_of_eval_event,
            tensorboard_run_directory=str(tensorboard_run_directory / "learner"),
        )

        for process in actor_process_list:
            process.terminate()

        return 0


if __name__ == "__main__":
    set_start_method("spawn")

    config = CEMRLTD3Config()
    main(config)
