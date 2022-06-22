"""
The code is in this file was adapted from https://github.com/deepmind/acme
"""

from copy import deepcopy
from typing import Callable, NamedTuple, Tuple

import haiku
import jax
import optax
import rlax
from jax import numpy as jnp

from fastpbrl.networks import AtariTorso, DuellingMLP
from fastpbrl.types import Action, CriticParams, Observation, Transition
from fastpbrl.utils import fix_transition_shape


class DQNHyperParameters(NamedTuple):
    learning_rate: float = 3e-4
    discount: float = 0.99
    huber_loss_parameter: float = 1.0
    target_update_period: int = 20


class DQNTrainingState(NamedTuple):
    critic_params: CriticParams
    target_critic_params: CriticParams
    critic_opt_state: optax.OptState
    steps: int
    random_key: jax.random.PRNGKey


class DQNNetworks(NamedTuple):
    critic_network: Callable[[CriticParams, Observation], jnp.ndarray]
    init_critic_network: Callable[[jax.random.PRNGKey], CriticParams]


def make_default_networks(
    num_frame_stack: int,
    image_shape: Tuple[int],
    num_actions: int,
    hidden_layer_sizes: Tuple[int, ...] = (256,),
) -> DQNNetworks:
    def _critic_forward_pass(obs: Observation) -> jnp.ndarray:
        critic_network = haiku.Sequential(
            [
                AtariTorso(),
                DuellingMLP(num_actions, hidden_sizes=hidden_layer_sizes),
            ]
        )
        return critic_network(obs)

    critic = haiku.without_apply_rng(haiku.transform(_critic_forward_pass))

    dummy_observation = jnp.expand_dims(
        jnp.zeros((*image_shape, num_frame_stack)), axis=0
    )

    return DQNNetworks(
        critic_network=critic.apply,
        init_critic_network=lambda key: critic.init(key, dummy_observation),
    )


def make_initial_training_state(
    networks: DQNNetworks,
    critic_optimizer: optax.GradientTransformation,
    random_key: jax.random.PRNGKey,
) -> DQNTrainingState:

    # Create the critic networks and optimizer
    critic_key, random_key = jax.random.split(random_key, 2)
    critic_params = networks.init_critic_network(critic_key)
    critic_opt_state = critic_optimizer.init(critic_params)
    target_critic_params = deepcopy(critic_params)

    return DQNTrainingState(
        critic_params=critic_params,
        target_critic_params=target_critic_params,
        critic_opt_state=critic_opt_state,
        steps=0,
        random_key=random_key,
    )


def select_action(
    critic_params: CriticParams,
    observation: Observation,
    networks: DQNNetworks,
) -> Action:
    action = networks.critic_network(critic_params, observation)
    return jnp.argmax(action, axis=-1)


def critic_loss(
    critic_params: CriticParams,
    state: DQNTrainingState,
    hyperparams: DQNHyperParameters,
    transition_batch: Transition,
    networks: DQNNetworks,
) -> jnp.ndarray:
    def get_q_value(q_value, action):
        return q_value[action]

    vmap_get_q_value = jax.vmap(get_q_value)

    # Forward pass.
    q_tm1 = networks.critic_network(critic_params, transition_batch.observation)
    q_t = networks.critic_network(
        state.target_critic_params, transition_batch.next_observation
    )

    # Compute Q-learning TD-error
    target_q_tm1 = transition_batch.reward + hyperparams.discount * (
        1.0 - transition_batch.done
    ) * jnp.max(q_t, axis=-1)

    td_error = jax.lax.stop_gradient(target_q_tm1) - vmap_get_q_value(
        q_tm1, transition_batch.action
    )
    batch_loss = rlax.huber_loss(td_error, hyperparams.huber_loss_parameter)

    return jnp.mean(batch_loss)


def critic_update_step(
    state: DQNTrainingState,
    hyperparams: DQNHyperParameters,
    transition_batch: Transition,
    networks: DQNNetworks,
    critic_optimizer: optax.GradientTransformation,
):

    critic_loss_and_grad = jax.value_and_grad(critic_loss)

    _, critic_gradients = critic_loss_and_grad(
        state.critic_params, state, hyperparams, transition_batch, networks
    )
    critic_updates, critic_opt_state = critic_optimizer.update(
        critic_gradients, state.critic_opt_state
    )
    critic_updates = jax.tree_map(
        lambda x: hyperparams.learning_rate * x, critic_updates
    )
    critic_params = optax.apply_updates(state.critic_params, critic_updates)

    target_critic_params = jax.lax.cond(
        state.steps % hyperparams.target_update_period == 0,
        lambda _: critic_params,
        lambda _: state.target_critic_params,
        operand=None,
    )

    return state._replace(
        critic_params=critic_params,
        target_critic_params=target_critic_params,
        critic_opt_state=critic_opt_state,
        steps=state.steps + 1,
        random_key=state.random_key,
    )


def update_step(
    state: DQNTrainingState,
    hyperparams: DQNHyperParameters,
    transition_batch: Transition,
    num_steps: int,
    networks: DQNNetworks,
    critic_optimizer: optax.GradientTransformation,
) -> DQNTrainingState:
    def one_update_step(
        intermediate_state: DQNTrainingState, one_step_transition_batch: Transition
    ) -> DQNTrainingState:

        intermediate_state = critic_update_step(
            intermediate_state,
            hyperparams,
            one_step_transition_batch,
            networks,
            critic_optimizer,
        )
        return intermediate_state, {}

    transition_batch = fix_transition_shape(transition_batch)

    return jax.lax.scan(one_update_step, state, transition_batch, length=num_steps)[0]
