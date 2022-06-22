from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

import fastpbrl.td3.cemrl as cemrl
from fastpbrl.td3.core import (
    TD3HyperParams,
    TD3Networks,
    TD3TrainingState,
    critic_update_step,
    fix_transition_shape,
    policy_loss,
)
from fastpbrl.types import Action, PolicyParams, Transition
from fastpbrl.utils import polyak_averaging

make_initial_training_state = cemrl.make_initial_training_state


class DVDTD3HyperParameters(NamedTuple):
    scale_rbf: float = 1.0  # radial basis function sigma used for scaling the squared
    # exponential kernel, see "Effective Diversity in Population Based Reinforcement
    # Learning", equation #5
    # https://github.com/jparkerholder/DvD_ES/blob/a5e6f66ce12c5690bfb7dc4d2b1686a3b98f2811/utils.py#L82
    novelty_weight: float = 1.0
    td3_params: TD3HyperParams = TD3HyperParams(delay=2)


def square_distance_matrix(vector_batch: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Euclidean distance matrix associated with a set of vectors.
    Specifically, for a set of N vectors, returns the NxN matrix where the
    entry in row i column j is the square euclidean distance between vector
    i and vector j, see https://en.wikipedia.org/wiki/Euclidean_distance_matrix

    Parameters:
        vector_batch (jnp.ndarray): expected shape is (N, M) where N is the number
            of vectors and M is the dimension of the vectors (i.e. vectors are
            stacked along the first axis).

    Return:
        A NxN square jnp.array
    """
    nb_vectors = vector_batch.shape[0]
    gram_matrix = jnp.dot(vector_batch, jnp.transpose(vector_batch))
    diag_gram_matrix = jnp.diagonal(gram_matrix)

    # Computes the square matrix efficiently using the Gram matrix
    # see https://en.wikipedia.org/wiki/Euclidean_distance_matrix
    return (
        jnp.dot(diag_gram_matrix.reshape((nb_vectors, 1)), jnp.ones((1, nb_vectors)))
        + jnp.dot(jnp.ones((nb_vectors, 1)), diag_gram_matrix.reshape(1, nb_vectors))
        - 2 * gram_matrix
    )


def novelty_loss(
    policy_params: PolicyParams,
    transitions: Transition,
    scale_rbf: float,
    networks: TD3Networks,
) -> jnp.ndarray:
    vmapped_policy_network = jax.vmap(networks.policy_network, in_axes=(0, None))
    actions = vmapped_policy_network(policy_params, transitions.observation)
    embeddings = jnp.reshape(actions, (actions.shape[0], -1))
    distance_matrix = square_distance_matrix(embeddings)

    return -jax.numpy.linalg.slogdet(jnp.exp(-distance_matrix / (2.0 * scale_rbf)))[1]


def policy_update_step(
    state: TD3TrainingState,
    hyperparams: DVDTD3HyperParameters,
    transitions: Transition,
    networks: TD3Networks,
    policy_optimizer: optax.GradientTransformation,
) -> TD3TrainingState:

    _polyak_averaging = partial(polyak_averaging, tau=hyperparams.td3_params.tau)
    policy_loss_and_grad = jax.value_and_grad(policy_loss)
    novelty_loss_and_grad = jax.value_and_grad(novelty_loss)

    vmapped_policy_loss_and_grad = jax.vmap(
        policy_loss_and_grad, in_axes=(0, None, 0, None)
    )
    vmapped_optimizer_update = jax.vmap(policy_optimizer.update)

    def _policy_update_step():
        _, policy_gradients = vmapped_policy_loss_and_grad(
            state.policy_params,
            state.critic_params,
            transitions.next_observation,
            networks,
        )

        _, novelty_gradients = novelty_loss_and_grad(
            state.policy_params, transitions, hyperparams.scale_rbf, networks
        )

        novelty_gradients = jax.tree_map(
            lambda x: hyperparams.novelty_weight * x, novelty_gradients
        )

        policy_gradients = jax.tree_multimap(
            lambda x, y: x + y, policy_gradients, novelty_gradients
        )

        policy_updates, policy_opt_state = vmapped_optimizer_update(
            policy_gradients, state.policy_opt_state
        )
        policy_updates = jax.tree_map(
            lambda x: hyperparams.td3_params.lr_policy * x, policy_updates
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
        state.steps % hyperparams.td3_params.delay == 0,
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
    hyperparams: DVDTD3HyperParameters,
    transition_batch: Transition,
    num_steps: int,
    networks: TD3Networks,
    critic_optimizer: optax.GradientTransformation,
    twin_critic_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    population_size: int,
):
    _vmapped_add_policy_noise = jax.vmap(
        networks.add_policy_noise, in_axes=(0, 0, None, None)
    )

    def vmapped_add_policy_noise(
        action: Action,
        random_key: jax.random.PRNGKey,
        target_sigma: jnp.ndarray,
        noise_clip: jnp.ndarray,
    ) -> Action:
        return _vmapped_add_policy_noise(
            action,
            jax.random.split(random_key, num=population_size),
            target_sigma,
            noise_clip,
        )

    vmapped_policy_networks = TD3Networks(
        policy_network=jax.vmap(networks.policy_network),
        critic_network=networks.critic_network,
        twin_critic_network=networks.twin_critic_network,
        add_policy_noise=vmapped_add_policy_noise,
        init_policy_network=networks.init_policy_network,
        init_critic_network=networks.init_critic_network,
        init_twin_critic_network=networks.init_twin_critic_network,
    )

    shared_critic_update_step = partial(
        critic_update_step,
        networks=vmapped_policy_networks,
        critic_optimizer=critic_optimizer,
        twin_critic_optimizer=twin_critic_optimizer,
    )

    _policy_update_step = partial(
        policy_update_step,
        networks=networks,
        policy_optimizer=policy_optimizer,
    )

    transition_batch = fix_transition_shape(transition_batch)

    def one_update_step(
        intermediate_state: TD3TrainingState, one_step_transition_batch: Transition
    ) -> TD3TrainingState:
        # Update the shared critic
        intermediate_state = shared_critic_update_step(
            intermediate_state,
            hyperparams.td3_params,
            one_step_transition_batch,
        )

        # Update the policies
        intermediate_state = _policy_update_step(
            intermediate_state,
            hyperparams,
            one_step_transition_batch,
        )

        return intermediate_state, {}

    return jax.lax.scan(one_update_step, state, transition_batch, length=num_steps)[0]
