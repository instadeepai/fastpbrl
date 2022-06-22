import itertools
from logging import Logger
from multiprocessing import Queue
from typing import Generator, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import tree
from gym import Env

from fastpbrl.types import (
    HyperParams,
    PolicyParams,
    SelectActionFunction,
    Step,
    Transition,
)


def step_generator(
    policy_params: PolicyParams,
    hyperparams: HyperParams,
    environment: Env,
    select_action_function: SelectActionFunction,
    max_steps_per_episode: int,
    num_warmup_steps: int = 0,
    transition_queue: Optional[Queue] = None,
    transition_queue_batch_size: int = 1,
    never_explore: bool = False,
    logger: Optional[Logger] = None,
) -> Generator[Step, Tuple[PolicyParams, HyperParams], None]:
    """
    Python generator (https://wiki.python.org/moin/Generators) to interact with an
    environment step by step, yielding on every step.
    This enables to:
    - seamlessly collect experience (in the form of Transition objects) that is pushed
      to a queue.
    - update the policy weights and policy hyper parameters on every step through the
       send function of the generator.

    Typical use:
    ```
    generator = step_generator(policy_params, hyperparams, ...)
    step = next(generator)  # initialization for the first step
    while True:
        step = generator.send((new_policy_params, new_hyperparams))
        ...
    ```

    Parameters:
        policy_params (PolicyParams): Initial value of the policy parameters to use
            for the first environment step
        hyperparams (HyperParams): Initial value of the policy parameters to use
            for the first environment step
        environment (Env): Gym environment to interact with
        select_action_function (SelectActionFunction): function that
            maps policy_params, hyperparams, and an obsevation to an action.
        max_steps_per_episode (int): Maximum number of times we step the environment
            per episode.
        num_warmup_steps (int): The first num_warmup_steps actions taken will be
            sampled randomly instead of using the policy.
        transition_queue (Optional[Queue]): If provided, collected transitions will
            be pushed to the queue.
        transition_queue_batch_size (int): for efficiency purposes, transitions
            will be pushed in batches of this size to the queue.
        never_explore (bool): If true, do not explore when interacting with the
            environment.
        logger (Optional[Logger]): If provided, episode returns will be logged.

    Return:
        A python generator which:
            - yields Step after each interaction with the environment
            - expects a 2-uple (PolicyParams, HyperParams)
    """

    action_min = environment.action_space.low.astype(np.float32)
    action_max = environment.action_space.high.astype(np.float32)

    pending_transitions_list = []
    total_step_id = 0
    for episode_id in itertools.count():
        episode_return = 0.0
        last_observation = environment.reset()

        for step_id in range(max_steps_per_episode):
            total_step_id += 1
            if total_step_id >= num_warmup_steps:
                action = select_action_function(
                    policy_params=policy_params,
                    hyperparams=hyperparams,
                    observation=last_observation,
                    exploration=not never_explore,
                )
            else:
                action = environment.action_space.sample().astype(np.float32)

            action = np.clip(action, action_min, action_max)

            observation, reward, is_episode_done, _ = environment.step(action)
            done = (
                1.0 if is_episode_done and step_id + 1 < max_steps_per_episode else 0.0
            )

            transition = Transition(
                observation=last_observation,
                action=action,
                reward=jnp.array([reward], dtype=np.float32),
                done=jnp.array([done], dtype=np.float32),
                next_observation=observation,
            )

            # Management of the transitions
            if transition_queue is not None:
                pending_transitions_list.append(transition)

                if len(pending_transitions_list) >= transition_queue_batch_size:
                    transition_queue.put(
                        tree.map_structure(
                            lambda *x: np.stack(x, axis=0),
                            *pending_transitions_list,
                        )
                    )
                    pending_transitions_list = []

            last_observation = observation
            episode_return += reward

            step = Step(
                total_step_id=total_step_id,
                episode_id=episode_id,
                step_id=step_id,
                current_return=episode_return,
                is_episode_done=is_episode_done,
                last_transition=transition,
            )

            policy_params, hyperparams = yield step

            if is_episode_done and logger is not None:
                logger.info(
                    f"Episode #{episode_id} (env steps={total_step_id}): "
                    f"return={episode_return:.2f} length={step_id + 1}"
                )

            # Logging of the results
            if is_episode_done:
                break
