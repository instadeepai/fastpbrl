"""
The code is in this file was adapted from https://github.com/deepmind/acme
"""

import functools
from typing import Callable, NamedTuple, Tuple

import gym
import haiku
import jax
import numpy as np
import optax
import rlax
from jax import numpy as jnp

from fastpbrl.networks import LayerNormMLP, NearZeroInitializedLinear, TanhToSpec
from fastpbrl.types import Action, CriticParams, Observation, PolicyParams, Transition
from fastpbrl.utils import fix_transition_shape, polyak_averaging


class TD3HyperParams(NamedTuple):
    lr_policy: float = 3e-4
    lr_critic: float = 3e-4
    discount: float = 0.99
    sigma: float = 0.1
    delay: int = 2
    target_sigma: float = 0.2
    noise_clip: float = 0.5
    tau: float = 0.005


class TD3TrainingState(NamedTuple):
    policy_params: PolicyParams
    target_policy_params: PolicyParams
    critic_params: CriticParams
    target_critic_params: CriticParams
    twin_critic_params: CriticParams
    target_twin_critic_params: CriticParams
    policy_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    twin_critic_opt_state: optax.OptState
    steps: int
    random_key: jax.random.PRNGKey


class TD3Networks(NamedTuple):
    policy_network: Callable[[PolicyParams, Observation], Action]
    critic_network: Callable[[CriticParams, Observation, Action], jnp.ndarray]
    twin_critic_network: Callable[[CriticParams, Observation, Action], jnp.ndarray]

    init_policy_network: Callable[[jax.random.PRNGKey], PolicyParams]
    init_critic_network: Callable[[jax.random.PRNGKey], CriticParams]
    init_twin_critic_network: Callable[[jax.random.PRNGKey], CriticParams]

    add_policy_noise: Callable[[Action, jax.random.PRNGKey, float, float], Action]


def make_default_networks(
    gym_env: gym.Env,
    hidden_layer_sizes: Tuple[int, ...] = (256, 256),
) -> TD3Networks:

    action_shape = gym_env.action_space.shape
    action_min = gym_env.action_space.low.astype(np.float32)
    action_max = gym_env.action_space.high.astype(np.float32)

    def add_policy_noise(
        action: Action,
        key: jax.random.PRNGKey,
        target_sigma: float,
        noise_clip: float,
    ) -> Action:
        """Adds action noise to bootstrapped Q-value estimate in critic loss."""
        noise = jax.random.normal(key=key, shape=action.shape) * target_sigma
        noise = jnp.clip(noise, -noise_clip, noise_clip)
        return jnp.clip(action + noise, action_min, action_max)

    def _policy_forward_pass(obs: Observation) -> Action:
        policy_network = haiku.Sequential(
            [
                LayerNormMLP(hidden_layer_sizes, activate_final=True),
                NearZeroInitializedLinear(np.prod(action_shape, dtype=int)),
                TanhToSpec(min_value=action_min, max_value=action_max),
            ]
        )
        return policy_network(obs)

    def _critic_forward_pass(obs: Observation, action: Action) -> jnp.ndarray:
        critic_network = haiku.Sequential(
            [
                LayerNormMLP(list(hidden_layer_sizes) + [1]),
            ]
        )
        input_ = jnp.concatenate([obs, action], axis=-1)
        value = critic_network(input_)
        return jnp.squeeze(value)

    policy = haiku.without_apply_rng(haiku.transform(_policy_forward_pass))
    critic = haiku.without_apply_rng(haiku.transform(_critic_forward_pass))

    # Create dummy observations and actions to create network parameters.
    dummy_action = jnp.expand_dims(
        jnp.zeros_like(gym_env.action_space.sample().astype(np.float32)), axis=0
    )
    dummy_obs = jax.tree_map(
        lambda x: jnp.expand_dims(jnp.zeros_like(x.astype(np.float32)), axis=0),
        gym_env.observation_space.sample(),
    )

    return TD3Networks(
        policy_network=policy.apply,
        critic_network=critic.apply,
        twin_critic_network=critic.apply,
        add_policy_noise=add_policy_noise,
        init_policy_network=lambda key: policy.init(key, dummy_obs),
        init_critic_network=lambda key: critic.init(key, dummy_obs, dummy_action),
        init_twin_critic_network=lambda key: critic.init(key, dummy_obs, dummy_action),
    )


def make_initial_training_state(
    networks: TD3Networks,
    critic_optimizer: optax.GradientTransformation,
    twin_critic_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    random_key: jax.random.PRNGKey,
) -> TD3TrainingState:

    (key_init_critic, key_init_twin_critic, key_actor, key_state) = jax.random.split(
        random_key, 4
    )

    # Create the network parameters and copy into the target network parameters.
    initial_critic_params = networks.init_critic_network(key_init_critic)
    initial_twin_critic_params = networks.init_twin_critic_network(key_init_twin_critic)
    initial_policy_params = networks.init_policy_network(key_actor)

    initial_target_policy_params = initial_policy_params
    initial_target_critic_params = initial_critic_params
    initial_target_twin_critic_params = initial_twin_critic_params

    # Initialize optimizers.
    initial_policy_opt_state = policy_optimizer.init(initial_policy_params)
    initial_critic_opt_state = critic_optimizer.init(initial_critic_params)
    initial_twin_critic_opt_state = twin_critic_optimizer.init(
        initial_twin_critic_params
    )

    return TD3TrainingState(
        policy_params=initial_policy_params,
        target_policy_params=initial_target_policy_params,
        critic_params=initial_critic_params,
        twin_critic_params=initial_twin_critic_params,
        target_critic_params=initial_target_critic_params,
        target_twin_critic_params=initial_target_twin_critic_params,
        policy_opt_state=initial_policy_opt_state,
        critic_opt_state=initial_critic_opt_state,
        twin_critic_opt_state=initial_twin_critic_opt_state,
        steps=0,
        random_key=key_state,
    )


def select_action(
    policy_params: PolicyParams,
    hyperparams: TD3HyperParams,
    observation: Observation,
    rng: jax.random.PRNGKey,
    exploration: bool,
    networks: TD3Networks,
) -> Tuple[Action, jax.random.PRNGKey]:
    action = networks.policy_network(policy_params, observation)
    if exploration:
        rng1, rng = jax.random.split(rng)
        noise = jax.random.normal(rng1, shape=action.shape) * hyperparams.sigma
        action += noise
    return action, rng


def critic_loss(
    critic_params: CriticParams,
    state: TD3TrainingState,
    hyperparams: TD3HyperParams,
    transition_batch: Transition,
    random_key: jax.random.PRNGKey,
    networks: TD3Networks,
) -> jnp.ndarray:
    q_tm1 = networks.critic_network(
        critic_params, transition_batch.observation, transition_batch.action
    )

    action = networks.policy_network(
        state.target_policy_params, transition_batch.next_observation
    )

    action_with_noise = networks.add_policy_noise(
        action, random_key, hyperparams.target_sigma, hyperparams.noise_clip
    )

    q_t = networks.critic_network(
        state.target_critic_params, transition_batch.next_observation, action_with_noise
    )
    twin_q_t = networks.twin_critic_network(
        state.target_twin_critic_params,
        transition_batch.next_observation,
        action_with_noise,
    )

    q_t = jnp.minimum(q_t, twin_q_t)

    target_q_tm1 = (
        transition_batch.reward
        + hyperparams.discount * (1.0 - transition_batch.done) * q_t
    )
    td_error = jax.lax.stop_gradient(target_q_tm1) - q_tm1

    return jnp.mean(jnp.square(td_error))


def critic_update_step(
    state: TD3TrainingState,
    hyperparams: TD3HyperParams,
    transition_batch: Transition,
    networks: TD3Networks,
    critic_optimizer: optax.GradientTransformation,
    twin_critic_optimizer: optax.GradientTransformation,
) -> TD3TrainingState:
    _polyak_averaging = functools.partial(polyak_averaging, tau=hyperparams.tau)
    critic_loss_and_grad = jax.value_and_grad(critic_loss)

    random_key, key_critic, key_twin = jax.random.split(state.random_key, 3)

    # Updates on the critic: compute the gradients, and update using
    # Polyak averaging.
    _, critic_gradients = critic_loss_and_grad(
        state.critic_params, state, hyperparams, transition_batch, key_critic, networks
    )
    critic_updates, critic_opt_state = critic_optimizer.update(
        critic_gradients, state.critic_opt_state
    )
    critic_updates = jax.tree_map(lambda x: hyperparams.lr_critic * x, critic_updates)
    critic_params = optax.apply_updates(state.critic_params, critic_updates)

    target_critic_params = jax.tree_multimap(
        _polyak_averaging, state.target_critic_params, critic_params
    )

    # Updates on the twin critic: compute the gradients, and update using
    # Polyak averaging.
    _, twin_critic_gradients = critic_loss_and_grad(
        state.twin_critic_params,
        state,
        hyperparams,
        transition_batch,
        key_twin,
        networks,
    )
    twin_critic_updates, twin_critic_opt_state = twin_critic_optimizer.update(
        twin_critic_gradients, state.twin_critic_opt_state
    )
    twin_critic_updates = jax.tree_map(
        lambda x: hyperparams.lr_critic * x, twin_critic_updates
    )
    twin_critic_params = optax.apply_updates(
        state.twin_critic_params, twin_critic_updates
    )
    target_twin_critic_params = jax.tree_multimap(
        _polyak_averaging,
        state.target_twin_critic_params,
        twin_critic_params,
    )

    return state._replace(
        critic_params=critic_params,
        target_critic_params=target_critic_params,
        twin_critic_params=twin_critic_params,
        target_twin_critic_params=target_twin_critic_params,
        critic_opt_state=critic_opt_state,
        twin_critic_opt_state=twin_critic_opt_state,
        random_key=random_key,
        steps=state.steps + 1,
    )


def policy_loss(
    policy_params: PolicyParams,
    critic_params: CriticParams,
    observation: Observation,
    networks: TD3Networks,
) -> jnp.ndarray:
    # Computes the discrete policy gradient loss.
    action = networks.policy_network(policy_params, observation)
    grad_critic = jax.vmap(
        jax.grad(networks.critic_network, argnums=2), in_axes=(None, 0, 0)
    )

    dq_da = grad_critic(critic_params, observation, action)
    batch_dpg_learning = jax.vmap(rlax.dpg_loss, in_axes=(0, 0))
    loss = batch_dpg_learning(action, dq_da)
    return jnp.mean(loss)


def policy_update_step(
    state: TD3TrainingState,
    hyperparams: TD3HyperParams,
    transition_batch: Transition,
    networks: TD3Networks,
    policy_optimizer: optax.GradientTransformation,
) -> TD3TrainingState:

    # Updates on the policy: compute the gradients, and update using
    # Polyak averaging (if delay enabled, the update might not be applied).

    _polyak_averaging = functools.partial(polyak_averaging, tau=hyperparams.tau)
    policy_loss_and_grad = jax.value_and_grad(policy_loss)

    def _policy_update_step():
        _, policy_gradients = policy_loss_and_grad(
            state.policy_params,
            state.critic_params,
            transition_batch.next_observation,
            networks,
        )

        policy_updates, policy_opt_state = policy_optimizer.update(
            policy_gradients, state.policy_opt_state
        )
        policy_updates = jax.tree_map(
            lambda x: hyperparams.lr_policy * x, policy_updates
        )
        policy_params = optax.apply_updates(state.policy_params, policy_updates)
        target_policy_params = jax.tree_multimap(
            _polyak_averaging, state.target_policy_params, policy_params
        )
        return policy_params, target_policy_params, policy_opt_state

    # The update on the policy is applied every `delay` steps.
    current_policy_state = (
        state.policy_params,
        state.target_policy_params,
        state.policy_opt_state,
    )
    policy_params, target_policy_params, policy_opt_state = jax.lax.cond(
        state.steps % hyperparams.delay == 0,
        lambda _: _policy_update_step(),
        lambda _: current_policy_state,
        operand=None,
    )

    return state._replace(
        policy_params=policy_params,
        target_policy_params=target_policy_params,
        policy_opt_state=policy_opt_state,
    )


def update_step(
    state: TD3TrainingState,
    hyperparams: TD3HyperParams,
    transition_batch: Transition,
    num_steps: int,
    networks: TD3Networks,
    critic_optimizer: optax.GradientTransformation,
    twin_critic_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
) -> TD3TrainingState:
    def one_update_step(
        intermediate_state: TD3TrainingState, one_step_transition_batch: Transition
    ) -> Tuple[TD3TrainingState, dict]:

        intermediate_state = critic_update_step(
            intermediate_state,
            hyperparams,
            one_step_transition_batch,
            networks,
            critic_optimizer,
            twin_critic_optimizer,
        )
        intermediate_state = policy_update_step(
            intermediate_state,
            hyperparams,
            one_step_transition_batch,
            networks,
            policy_optimizer,
        )
        return intermediate_state, {}

    transition_batch = fix_transition_shape(transition_batch)

    return jax.lax.scan(one_update_step, state, transition_batch, length=num_steps)[0]
