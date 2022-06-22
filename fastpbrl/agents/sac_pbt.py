import random
from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import loguniform

from fastpbrl import utils
from fastpbrl.agents.abstract import PBTAgent
from fastpbrl.sac.core import (
    SACHyperParams,
    SACTrainingState,
    make_default_networks,
    make_initial_training_state,
    select_action,
    update_step,
)


class SACPBT(PBTAgent):
    def __init__(
        self,
        gym_env_name: str,
        seed: int,
        population_size: int,
        hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    ):
        super(SACPBT, self).__init__(
            population_size=population_size, random_key=jax.random.PRNGKey(seed)
        )

        self.seed = seed
        self.population_size = population_size

        self._gym_env_name = gym_env_name
        self._hidden_layer_sizes = hidden_layer_sizes
        environment = utils.make_gym_env(gym_env_name)

        self._networks = make_default_networks(
            gym_env=environment, hidden_layer_sizes=self._hidden_layer_sizes
        )

        self._select_action = jax.jit(
            partial(
                select_action,
                networks=self._networks,
                min_action=jnp.asarray(environment.action_space.low),
                max_action=jnp.asarray(environment.action_space.high),
            ),
            static_argnames="exploration",
        )

        self._critic_optimizer = optax.adam(learning_rate=1.0)
        self._policy_optimizer = optax.adam(learning_rate=1.0)
        self._alpha_optimizer = optax.adam(learning_rate=1.0)

        _update_step = jax.jit(
            partial(
                update_step,
                networks=self._networks,
                critic_optimizer=self._critic_optimizer,
                policy_optimizer=self._policy_optimizer,
                alpha_optimizer=self._alpha_optimizer,
            ),
            static_argnames="num_steps",
        )

        self._update_step = jax.jit(
            jax.vmap(_update_step, in_axes=(0, 0, 0, None)),
            static_argnums=3,
        )

    def _make_initial_hyperparams(self) -> SACHyperParams:
        environment = utils.make_gym_env(self._gym_env_name)
        return SACHyperParams(
            lr_critic=loguniform.rvs(3e-5, 3e-3),
            lr_policy=loguniform.rvs(3e-5, 3e-3),
            lr_alpha=loguniform.rvs(3e-5, 3e-3),
            adaptive_entropy_coefficient=True,
            target_entropy=-np.prod(environment.action_space.shape, dtype=float),
            reward_scale=loguniform.rvs(0.1, 10.0),
            discount=random.uniform(0.9, 1.0),
        )

    def _make_initial_training_state(self) -> SACTrainingState:
        self._random_key, random_key = jax.random.split(self._random_key)
        return make_initial_training_state(
            networks=self._networks,
            critic_optimizer=self._critic_optimizer,
            alpha_optimizer=self._alpha_optimizer,
            policy_optimizer=self._policy_optimizer,
            random_key=random_key,
        )

    def reset_optimizers(
        self, training_state: SACTrainingState, index_list: List[int]
    ) -> SACTrainingState:
        state_list = utils.unpack(training_state)
        initial_training_state = self._make_initial_training_state()
        for index in index_list:
            state_list[index] = state_list[index]._replace(
                policy_opt_state=initial_training_state.policy_opt_state,
                alpha_opt_state=initial_training_state.alpha_opt_state,
                alpha_params=initial_training_state.alpha_params,
            )
        return utils.jax_tree_stack(state_list)
