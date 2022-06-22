import random
from functools import partial
from typing import List, Tuple

import jax
import optax
from scipy.stats import loguniform

from fastpbrl import utils
from fastpbrl.agents.abstract import PBTAgent
from fastpbrl.td3.core import (
    TD3HyperParams,
    TD3TrainingState,
    make_default_networks,
    make_initial_training_state,
    select_action,
    update_step,
)


class TD3PBT(PBTAgent):
    def __init__(
        self,
        gym_env_name: str,
        seed: int,
        population_size: int,
        hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    ):
        super(TD3PBT, self).__init__(
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
            ),
            static_argnames="exploration",
        )

        self._critic_optimizer = optax.adam(learning_rate=1.0)
        self._twin_critic_optimizer = optax.adam(learning_rate=1.0)
        self._policy_optimizer = optax.adam(learning_rate=1.0)

        _update_step = jax.jit(
            partial(
                update_step,
                networks=self._networks,
                critic_optimizer=self._critic_optimizer,
                twin_critic_optimizer=self._twin_critic_optimizer,
                policy_optimizer=self._policy_optimizer,
            ),
            static_argnames="num_steps",
        )

        self._update_step = jax.jit(
            jax.vmap(_update_step, in_axes=(0, 0, 0, None)),
            static_argnums=3,
        )

    def _make_initial_hyperparams(self) -> TD3HyperParams:
        return TD3HyperParams(
            lr_critic=loguniform.rvs(3e-5, 3e-3),
            lr_policy=loguniform.rvs(3e-5, 3e-3),
            delay=random.randint(1, 5),
            sigma=random.uniform(0.0, 1.0),
            target_sigma=random.uniform(0.0, 1.0),
            discount=random.uniform(0.9, 1.0),
            noise_clip=random.uniform(0.0, 1.0),
        )

    def _make_initial_training_state(self) -> TD3TrainingState:
        self._random_key, random_key = jax.random.split(self._random_key)
        return make_initial_training_state(
            networks=self._networks,
            critic_optimizer=self._critic_optimizer,
            twin_critic_optimizer=self._twin_critic_optimizer,
            policy_optimizer=self._policy_optimizer,
            random_key=random_key,
        )

    def reset_optimizers(
        self, training_state: TD3TrainingState, index_list: List[int]
    ) -> TD3TrainingState:
        state_list = utils.unpack(training_state)
        initial_training_state = self._make_initial_training_state()
        for index in index_list:
            state_list[index] = state_list[index]._replace(
                target_policy_params=state_list[index].policy_params,
                policy_opt_state=initial_training_state.policy_opt_state,
            )
        return utils.jax_tree_stack(state_list)
