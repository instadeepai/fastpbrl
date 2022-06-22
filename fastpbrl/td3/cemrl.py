from copy import deepcopy
from functools import partial

import jax
import optax

from fastpbrl.td3.core import (
    TD3HyperParams,
    TD3Networks,
    TD3TrainingState,
    critic_update_step,
    fix_transition_shape,
    policy_update_step,
)
from fastpbrl.types import Transition
from fastpbrl.utils import jax_tree_stack


def make_initial_training_state(
    networks: TD3Networks,
    critic_optimizer: optax.GradientTransformation,
    twin_critic_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    random_key: jax.random.PRNGKey,
    population_size: int,
) -> TD3TrainingState:

    # Create the critic networks and optimizer
    critic_key, twin_critic_key, random_key = jax.random.split(random_key, 3)

    critic_params = networks.init_critic_network(critic_key)
    twin_critic_params = networks.init_twin_critic_network(twin_critic_key)

    target_critic_params = deepcopy(critic_params)
    target_twin_critic_params = deepcopy(twin_critic_params)

    critic_opt_state = critic_optimizer.init(critic_params)
    twin_critic_opt_state = twin_critic_optimizer.init(twin_critic_params)

    # Create the policy networks and optimizer
    *policy_key_list, random_key = jax.random.split(random_key, population_size + 1)
    policy_params_list = [networks.init_policy_network(k) for k in policy_key_list]
    policy_opt_state_list = [policy_optimizer.init(p) for p in policy_params_list]

    stacked_policy_params = jax_tree_stack(policy_params_list)
    stacked_policy_opt_state = jax_tree_stack(policy_opt_state_list)
    stacked_target_policy_params = deepcopy(stacked_policy_params)

    return TD3TrainingState(
        policy_params=stacked_policy_params,
        target_policy_params=stacked_target_policy_params,
        critic_params=critic_params,
        twin_critic_params=twin_critic_params,
        target_critic_params=target_critic_params,
        target_twin_critic_params=target_twin_critic_params,
        policy_opt_state=stacked_policy_opt_state,
        critic_opt_state=critic_opt_state,
        twin_critic_opt_state=twin_critic_opt_state,
        steps=0,
        random_key=random_key,
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
    population_size: int,
):
    _vmap_func_add_policy_noise = jax.vmap(
        networks.add_policy_noise, in_axes=(0, 0, None, None)
    )

    def vmapped_add_policy_noise(action, random_key, target_sigma, noise_clip):
        return _vmap_func_add_policy_noise(
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

    single_policy_update_step = partial(
        policy_update_step,
        networks=networks,
        policy_optimizer=policy_optimizer,
    )

    vmap_axes_training_state_step_policy = TD3TrainingState(
        policy_params=0,
        target_policy_params=0,
        critic_params=None,
        target_critic_params=None,
        twin_critic_params=None,
        target_twin_critic_params=None,
        policy_opt_state=0,
        critic_opt_state=None,
        twin_critic_opt_state=None,
        steps=None,
        random_key=0,
    )

    all_policies_update_step = jax.vmap(
        single_policy_update_step,
        in_axes=(
            vmap_axes_training_state_step_policy,
            None,
            0,
        ),
        out_axes=vmap_axes_training_state_step_policy,
    )

    def one_update_step(
        intermediate_state: TD3TrainingState, one_step_transition_batch: Transition
    ) -> TD3TrainingState:

        intermediate_state = shared_critic_update_step(
            intermediate_state, hyperparams, one_step_transition_batch
        )
        random_key, updated_random_key = jax.random.split(
            intermediate_state.random_key, 2
        )
        intermediate_state = all_policies_update_step(
            intermediate_state._replace(
                random_key=jax.random.split(random_key, population_size)
            ),
            hyperparams,
            one_step_transition_batch,
        )
        intermediate_state = intermediate_state._replace(random_key=updated_random_key)

        return intermediate_state, {}

    transition_batch = fix_transition_shape(transition_batch)

    return jax.lax.scan(one_update_step, state, transition_batch, length=num_steps)[0]
