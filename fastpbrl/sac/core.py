"""
The code is in this file was adapted from https://github.com/deepmind/acme
"""

import functools
import random
from typing import Callable, NamedTuple, Optional, Tuple

import gym
import haiku
import jax
import numpy as np
import optax
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from fastpbrl.networks import NormalTanhDistribution
from fastpbrl.types import (
    Action,
    CriticParams,
    EntropyParams,
    LogProbFn,
    Observation,
    PolicyParams,
    PRNGKey,
    SampleFn,
    Transition,
)
from fastpbrl.utils import fix_transition_shape, polyak_averaging


class SACHyperParams(NamedTuple):
    lr_policy: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    tau: float = 0.005
    reward_scale: float = 1.0
    discount: float = 0.99
    adaptive_entropy_coefficient: bool = True
    entropy_coefficient: float = 0.0
    target_entropy: float = 0.0


class SACTrainingState(NamedTuple):
    """Contains training state for the learner."""

    policy_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    policy_params: PolicyParams
    critic_params: CriticParams
    target_critic_params: CriticParams
    random_key: PRNGKey
    alpha_opt_state: Optional[optax.OptState] = None
    alpha_params: Optional[EntropyParams] = None


class SACNetworks(NamedTuple):
    """Network and pure functions for the SAC agent.."""

    policy_network: Callable[[PolicyParams, Observation], tfd.Distribution]
    critic_network: Callable[[CriticParams, Observation, Action], jnp.ndarray]

    init_policy_network: Callable[[PRNGKey], PolicyParams]
    init_critic_network: Callable[[PRNGKey], CriticParams]

    log_prob: LogProbFn
    sample: SampleFn
    sample_eval: Optional[SampleFn] = None


def make_dummy_policy_params(
    networks: SACNetworks, random_key: PRNGKey
) -> PolicyParams:
    if random_key is None:
        random_key = PRNGKey(random.randint(0, 1 << 30))
    return make_initial_training_state(
        networks=networks,
        critic_optimizer=optax.adam(learning_rate=1.0),
        alpha_optimizer=optax.adam(learning_rate=1.0),
        policy_optimizer=optax.adam(learning_rate=1.0),
        random_key=random_key,
    ).policy_params


def make_initial_training_state(
    networks: SACNetworks,
    critic_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    alpha_optimizer: optax.GradientTransformation,
    random_key: PRNGKey,
) -> SACTrainingState:
    (key_init_critic, key_actor, key_state) = jax.random.split(random_key, 3)

    # Create the network parameters and copy into the target network parameters.
    initial_critic_params = networks.init_critic_network(key_init_critic)
    initial_policy_params = networks.init_policy_network(key_actor)
    initial_target_critic_params = initial_critic_params

    # Initialize optimizers.
    initial_policy_opt_state = policy_optimizer.init(initial_policy_params)
    initial_critic_opt_state = critic_optimizer.init(initial_critic_params)

    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    training_state = SACTrainingState(
        policy_opt_state=initial_policy_opt_state,
        critic_opt_state=initial_critic_opt_state,
        policy_params=initial_policy_params,
        critic_params=initial_critic_params,
        target_critic_params=initial_target_critic_params,
        random_key=key_state,
        alpha_params=log_alpha,
        alpha_opt_state=alpha_optimizer_state,
    )
    return training_state


def apply_policy_and_sample(networks: SACNetworks, eval_mode: bool = False):
    """Returns a function that computes actions."""
    sample_fn = networks.sample if not eval_mode else networks.sample_eval
    if not sample_fn:
        raise ValueError("sample function is not provided")

    def apply_and_sample(params, key, obs):
        return sample_fn(networks.policy_network(params, obs), key)

    return apply_and_sample


def make_default_networks(
    gym_env: gym.Env, hidden_layer_sizes: Tuple[int, ...] = (256, 256)
) -> SACNetworks:
    """Creates networks used by the agent."""

    num_dimensions = int(np.prod(gym_env.action_space.shape, dtype=int))

    def _actor_fn(obs):
        network = haiku.Sequential(
            [
                haiku.nets.MLP(
                    list(hidden_layer_sizes),
                    w_init=haiku.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                    activate_final=True,
                ),
                NormalTanhDistribution(num_dimensions),
            ]
        )
        return network(obs)

    def _critic_fn(obs, action):
        network1 = haiku.Sequential(
            [
                haiku.nets.MLP(
                    list(hidden_layer_sizes) + [1],
                    w_init=haiku.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                ),
            ]
        )
        network2 = haiku.Sequential(
            [
                haiku.nets.MLP(
                    list(hidden_layer_sizes) + [1],
                    w_init=haiku.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                ),
            ]
        )
        input_ = jnp.concatenate([obs, action], axis=-1)
        value1 = network1(input_)
        value2 = network2(input_)
        return jnp.concatenate([value1, value2], axis=-1)

    policy = haiku.without_apply_rng(haiku.transform(_actor_fn, apply_rng=True))
    critic = haiku.without_apply_rng(haiku.transform(_critic_fn, apply_rng=True))

    # Create dummy observations and actions to create network parameters.
    dummy_action = jnp.expand_dims(
        jnp.zeros_like(gym_env.action_space.sample().astype(np.float32)), axis=0
    )
    dummy_obs = jax.tree_map(
        lambda x: jnp.expand_dims(jnp.zeros_like(x.astype(np.float32)), axis=0),
        gym_env.observation_space.sample(),
    )

    return SACNetworks(
        policy_network=policy.apply,
        critic_network=critic.apply,
        init_critic_network=lambda key: critic.init(key, dummy_obs, dummy_action),
        init_policy_network=lambda key: policy.init(key, dummy_obs),
        log_prob=lambda params, actions: params.log_prob(actions),
        sample=lambda params, key: params.sample(seed=key),
        sample_eval=lambda params, key: params.mode(),
    )


def select_action(
    policy_params: PolicyParams,
    observation: Observation,
    rng: PRNGKey,
    exploration: bool,
    hyperparams: SACHyperParams,  # noqa to have the same signature as TD3
    networks: SACNetworks,
    min_action: jnp.DeviceArray,
    max_action: jnp.DeviceArray,
):
    rng1, rng2 = jax.random.split(rng)

    dist_params = networks.policy_network(policy_params, observation)
    if exploration:
        action = dist_params.sample(seed=rng1)
    else:
        action = dist_params.mode()
    return jnp.clip(action, min_action, max_action), rng2


def alpha_loss(
    log_alpha: jnp.ndarray,
    policy_params: PolicyParams,
    params: SACHyperParams,
    transitions: Transition,
    key: PRNGKey,
    networks: SACNetworks,
) -> jnp.ndarray:
    """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
    dist_params = networks.policy_network(policy_params, transitions.observation)
    action = networks.sample(dist_params, key)
    log_prob = networks.log_prob(dist_params, action)
    alpha = jnp.exp(log_alpha)
    alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - params.target_entropy)
    return jnp.mean(alpha_loss)


def critic_loss(
    critic_params: CriticParams,
    policy_params: PolicyParams,
    target_critic_params: CriticParams,
    alpha: jnp.ndarray,
    hyperparams: SACHyperParams,
    transitions: Transition,
    key: PRNGKey,
    networks: SACNetworks,
) -> jnp.ndarray:
    q_old_action = networks.critic_network(
        critic_params, transitions.observation, transitions.action
    )
    next_dist_params = networks.policy_network(
        policy_params, transitions.next_observation
    )
    next_action = networks.sample(next_dist_params, key)
    next_log_prob = networks.log_prob(next_dist_params, next_action)
    next_q = networks.critic_network(
        target_critic_params, transitions.next_observation, next_action
    )
    next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
    target_q = jax.lax.stop_gradient(
        transitions.reward * hyperparams.reward_scale
        + hyperparams.discount * hyperparams.discount * next_v
    )
    q_error = q_old_action - jnp.expand_dims(target_q, -1)
    q_loss = 0.5 * jnp.mean(jnp.square(q_error))
    return q_loss


def policy_loss(
    policy_params: PolicyParams,
    critic_params: CriticParams,
    alpha: jnp.ndarray,
    transitions: Transition,
    random_key: PRNGKey,
    networks: SACNetworks,
) -> jnp.ndarray:
    dist_params = networks.policy_network(policy_params, transitions.observation)
    action = networks.sample(dist_params, random_key)
    log_prob = networks.log_prob(dist_params, action)
    q_action = networks.critic_network(critic_params, transitions.observation, action)
    min_q = jnp.min(q_action, axis=-1)
    actor_loss = alpha * log_prob - min_q
    return jnp.mean(actor_loss)


def critic_update_step(
    state: SACTrainingState,
    hyperparams: SACHyperParams,
    transitions: Transition,
    networks: SACNetworks,
    critic_optimizer: optax.GradientTransformation,
) -> SACTrainingState:
    _polyak_averaging = functools.partial(polyak_averaging, tau=hyperparams.tau)
    critic_loss_and_grad = jax.value_and_grad(critic_loss)

    random_key, key_critic = jax.random.split(state.random_key, 2)

    alpha = jax.lax.cond(
        hyperparams.adaptive_entropy_coefficient,
        lambda _: jnp.exp(state.alpha_params),
        lambda _: hyperparams.entropy_coefficient,
        operand=None,
    )
    _, critic_grads = critic_loss_and_grad(
        state.critic_params,
        state.policy_params,
        state.target_critic_params,
        alpha,
        hyperparams,
        transitions,
        key_critic,
        networks,
    )

    # Apply critic gradients
    critic_updates, critic_opt_state = critic_optimizer.update(
        critic_grads, state.critic_opt_state
    )
    critic_updates = jax.tree_map(lambda x: hyperparams.lr_critic * x, critic_updates)
    critic_params = optax.apply_updates(state.critic_params, critic_updates)

    target_critic_params = jax.tree_multimap(
        _polyak_averaging, state.target_critic_params, critic_params
    )

    return state._replace(
        critic_opt_state=critic_opt_state,
        critic_params=critic_params,
        target_critic_params=target_critic_params,
        random_key=random_key,
    )


def policy_update_step(
    state: SACTrainingState,
    hyperparams: SACHyperParams,
    transition_batch: Transition,
    networks: SACNetworks,
    policy_optimizer: optax.GradientTransformation,
) -> SACTrainingState:

    policy_loss_and_grad = jax.value_and_grad(policy_loss)
    random_key, policy_key = jax.random.split(state.random_key, 2)

    alpha = jax.lax.cond(
        hyperparams.adaptive_entropy_coefficient,
        lambda _: jnp.exp(state.alpha_params),
        lambda _: hyperparams.entropy_coefficient,
        operand=None,
    )

    _, policy_gradients = policy_loss_and_grad(
        state.policy_params,
        state.critic_params,
        alpha,
        transition_batch,
        policy_key,
        networks,
    )

    # Apply policy gradients
    policy_updates, policy_opt_state = policy_optimizer.update(
        policy_gradients, state.policy_opt_state
    )
    policy_updates = jax.tree_map(lambda x: hyperparams.lr_policy * x, policy_updates)
    policy_params = optax.apply_updates(state.policy_params, policy_updates)

    return state._replace(
        policy_opt_state=policy_opt_state,
        policy_params=policy_params,
        random_key=random_key,
    )


def alpha_update_step(
    state: SACTrainingState,
    hyperparams: SACHyperParams,
    transitions: Transition,
    networks: SACNetworks,
    alpha_optimizer: optax.GradientTransformation,
) -> SACTrainingState:

    alpha_loss_and_grad = jax.value_and_grad(alpha_loss)
    random_key, alpha_key = jax.random.split(state.random_key, 2)

    _, alpha_grads = alpha_loss_and_grad(
        state.alpha_params,
        state.policy_params,
        hyperparams,
        transitions,
        alpha_key,
        networks,
    )

    alpha_updates, alpha_opt_state = alpha_optimizer.update(
        alpha_grads, state.alpha_opt_state
    )
    alpha_updates = jax.tree_map(lambda x: hyperparams.lr_alpha * x, alpha_updates)

    alpha_params = optax.apply_updates(state.alpha_params, alpha_updates)

    return state._replace(
        alpha_opt_state=alpha_opt_state,
        alpha_params=alpha_params,
        random_key=random_key,
    )


def update_step(
    state: SACTrainingState,
    hyperparams: SACHyperParams,
    transition_batch: Transition,
    num_steps: int,
    networks: SACNetworks,
    policy_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
    alpha_optimizer: optax.GradientTransformation,
):
    def one_update_step(
        intermediate_state: SACTrainingState, one_step_transition_batch: Transition
    ) -> Tuple[SACTrainingState, dict]:

        intermediate_state = critic_update_step(
            state=intermediate_state,
            hyperparams=hyperparams,
            transitions=one_step_transition_batch,
            networks=networks,
            critic_optimizer=critic_optimizer,
        )
        intermediate_state = policy_update_step(
            state=intermediate_state,
            hyperparams=hyperparams,
            transition_batch=one_step_transition_batch,
            networks=networks,
            policy_optimizer=policy_optimizer,
        )
        intermediate_state = jax.lax.cond(
            hyperparams.adaptive_entropy_coefficient,
            lambda _: alpha_update_step(
                state=intermediate_state,
                hyperparams=hyperparams,
                transitions=one_step_transition_batch,
                networks=networks,
                alpha_optimizer=alpha_optimizer,
            ),
            lambda _: intermediate_state,
            operand=None,
        )

        return intermediate_state, {}

    transition_batch = fix_transition_shape(transition_batch)

    return jax.lax.scan(one_update_step, state, transition_batch, length=num_steps)[0]
