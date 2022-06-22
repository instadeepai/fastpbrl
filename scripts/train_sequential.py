import argparse
import datetime
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import gym
import jax
import jax.numpy as jnp
import numpy as np
import tree
from tensorboardX import SummaryWriter

from fastpbrl.agents.sac import SAC
from fastpbrl.agents.td3 import TD3
from fastpbrl.evaluate import evaluate
from fastpbrl.replay_buffer import ReplayBuffer
from fastpbrl.sac.core import SACHyperParams
from fastpbrl.step_generator import step_generator
from fastpbrl.td3.core import TD3HyperParams
from fastpbrl.utils import make_gym_env, write_config

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)


@dataclass
class SequentialConfig:
    env_name: str = "HalfCheetah-v2"  # Gym environment name
    seed: Optional[int] = None  # Seed used for reproducibility
    # Total number of environment steps performed by the agent
    total_num_env_steps: int = 1_000_000  # Training ends when this total number
    # of environment steps have been carried out jointly by all of the actors
    use_gpu: bool = True  # If True, the learner looks for and uses a gpu to carry
    # out the update steps
    batch_size: int = 256  # Batch size for computing losses and gradients
    num_warmup_env_steps: int = 2_000  # The first num_warmup_env_steps actions taken
    # by the actors will be sampled randomly instead of using the policy
    num_update_step_at_once: int = 50  # Frequency, in number of update steps, at
    # which the learner updates the actors' with the latest policy parameters
    evaluation_frequency: int = 10_000  # Evaluate the policy every evaluation_frequency
    # environment steps carried out jointly by all actors
    num_episodes_per_evaluation: int = 10  # Number of episodes for each evaluation
    hyperparams: Union[SACHyperParams, TD3HyperParams] = TD3HyperParams()
    hidden_layer_sizes: Tuple[int] = (256, 256)  # Number and size of the hidden layers
    # of the policy and critic networks
    update_step_per_env_step: float = 1.0  # Target ratio of the number of update steps
    # per environment step
    min_size_to_sample: int = 2_000  # Minimum number of transitions in the replay
    # buffer to allow sampling from it.
    max_replay_buffer_size: int = 1_000_000  # Maximum number of transitions stored in
    # the replay buffer, transitions are deleted in a FIFO fashion afterwards

    def __post_init__(self):
        self.seed = self.seed if self.seed is not None else random.randint(0, 1_000_000)
        if isinstance(self.hyperparams, SACHyperParams):
            self.agent = SAC
        elif isinstance(self.hyperparams, TD3HyperParams):
            self.agent = TD3
        else:
            raise NotImplementedError('Only "TD3" and "SAC" are implemented')


def main(config: SequentialConfig):
    # Logging initialization
    logger = logging.getLogger()

    # Set-up the run name for tensorboard as well as the tensorboard logs directories
    tensorboard_run_name = (
        f'sequential_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    tensorboard_run_directory = str(
        Path(__file__).parent.parent.joinpath("logs").joinpath(tensorboard_run_name)
    )
    print(f"tensorboard --logdir {tensorboard_run_directory} --port={str(6006)}")
    tensorboard_writer = SummaryWriter(
        tensorboard_run_directory, flush_secs=1, write_to_disk=True
    )
    write_config(tensorboard_writer, config)

    start_time = time.time()

    # Determine jax device to use (gpu or cpu)
    device = jax.devices(backend="gpu" if config.use_gpu else "cpu")[0]

    # Environments initializations
    environment = make_gym_env(config.env_name)
    eval_environment = make_gym_env(config.env_name)
    max_steps_per_episode = gym.make(config.env_name)._max_episode_steps  # noqa

    # Replay buffer initialization
    replay_buffer = ReplayBuffer(max_size=config.max_replay_buffer_size)

    # Agent initialization
    agent = config.agent(
        gym_env_name=config.env_name,
        seed=config.seed,
        hidden_layer_sizes=config.hidden_layer_sizes,
    )

    # Initialize the neural network weights / optimizer states
    training_state = agent.make_initial_training_state()
    training_state = jax.device_put(training_state, device)

    # Initialization of the step generator
    env_step_generator = step_generator(
        policy_params=training_state.policy_params,
        hyperparams=config.hyperparams,
        environment=environment,
        num_warmup_steps=config.num_warmup_env_steps,
        max_steps_per_episode=max_steps_per_episode,
        select_action_function=agent.select_action,
        logger=logger,
    )
    step = next(env_step_generator)

    # Training loop
    update_step_id = 0
    for global_env_step_id in range(config.total_num_env_steps):

        # Step the environment using the latest policy parameters available
        step = env_step_generator.send(
            (training_state.policy_params, config.hyperparams)
        )

        # Add the transition to the replay buffer
        replay_buffer.add(
            tree.map_structure(
                lambda x: np.expand_dims(x, axis=0), step.last_transition
            )
        )

        # Carry out update steps if required
        if (
            global_env_step_id >= config.min_size_to_sample
            and update_step_id
            < (global_env_step_id - config.min_size_to_sample)
            * config.update_step_per_env_step
        ):
            update_step_id += config.num_update_step_at_once

            start_time_fetch_batch = time.perf_counter()

            # Fetch num_update_step_at_once * batch_size transitions et re-organize them
            # to be able to carry out num_update_step_at_once update steps at once
            transition_batch = replay_buffer.sample(
                config.batch_size * config.num_update_step_at_once
            )
            shape = (config.num_update_step_at_once, config.batch_size, -1)
            transition_batch = tree.map_structure(
                lambda x: jax.device_put(jnp.reshape(x, shape), device),
                transition_batch,
            )

            start_time_update_step = time.perf_counter()
            training_state = agent.update_step(
                training_state,
                hyperparams=config.hyperparams,
                transition_batch=transition_batch,
                num_steps=config.num_update_step_at_once,
            )

            tensorboard_writer.add_scalars(
                main_tag="elapsed_time/learner",
                tag_scalar_dict={
                    "fetch_batch": start_time_update_step - start_time_fetch_batch,
                    "train_step": (time.perf_counter() - start_time_update_step),
                },
                global_step=global_env_step_id,
                walltime=time.time() - start_time,
            )

        # Evaluate the current policy parameters if necessary
        if global_env_step_id % config.evaluation_frequency == 0:
            mean_return = np.mean(
                [
                    evaluate(
                        environment=eval_environment,
                        select_action_function=agent.select_action,
                        policy_params=training_state.policy_params,
                        hyperparams=config.hyperparams,
                    )
                    for _ in range(config.num_episodes_per_evaluation)
                ]
            )

            tensorboard_writer.add_scalar(
                tag="evaluation/episode_return",
                scalar_value=mean_return,
                global_step=global_env_step_id,
                walltime=time.time() - start_time,
            )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-sac",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    config = SequentialConfig(
        hyperparams=SACHyperParams() if args.use_sac else TD3HyperParams()
    )
    main(config)
