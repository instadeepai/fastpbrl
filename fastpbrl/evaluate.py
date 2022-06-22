import gym
import numpy as np

from fastpbrl.types import HyperParams, PolicyParams, SelectActionFunction


def evaluate(
    environment: gym.Env,
    select_action_function: SelectActionFunction,
    policy_params: PolicyParams,
    hyperparams: HyperParams,
) -> float:

    """
    Evaluate a policy defined by a combination of:
        - the policy weights
        - the policy hyperparameters
        - and a function that maps an observation to an action
    over an episode of the specified gym environment

    Parameters:
        environment (gym.Env): Gym environment to use for evaluation
        select_action_function (SelectActionFunction): function that
            maps policy_params, hyperparams, and an obsevation to an action.
        policy_params (PolicyParams): policy weights to use for evaluation
        hyperparams (HyperParams): hyperparameters of the policy to use for
            evaluation

    Return:
        Sum of rewards collected during an episode-worth of interactions with
        the environment.
    """
    action_min = environment.action_space.low.astype(np.float32)
    action_max = environment.action_space.high.astype(np.float32)

    episode_return = 0.0
    last_observation = environment.reset()
    while True:
        action = select_action_function(
            policy_params=policy_params,
            hyperparams=hyperparams,
            observation=last_observation,
            exploration=False,
        )

        action = np.clip(action, action_min, action_max)

        last_observation, reward, is_episode_done, _ = environment.step(action)

        episode_return += reward

        if is_episode_done:
            return episode_return
