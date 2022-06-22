from functools import partial
from typing import Tuple

import jax
import optax

from fastpbrl.agents.abstract import Agent
from fastpbrl.td3.cemrl import make_initial_training_state, update_step
from fastpbrl.td3.core import make_default_networks, select_action
from fastpbrl.types import (
    Action,
    HyperParams,
    Observation,
    PolicyParams,
    TrainingState,
    Transition,
)
from fastpbrl.utils import make_gym_env


class TD3CEMRL(Agent):
    def __init__(
        self,
        gym_env_name: str,
        population_size: int,
        seed: int,
        hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    ):
        self.population_size = population_size
        self.seed = seed
        self.hidden_layer_sizes = hidden_layer_sizes

        self._networks = make_default_networks(
            make_gym_env(gym_env_name), self.hidden_layer_sizes
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

        self._update_step = jax.jit(
            partial(
                update_step,
                networks=self._networks,
                critic_optimizer=self._critic_optimizer,
                twin_critic_optimizer=self._twin_critic_optimizer,
                policy_optimizer=self._policy_optimizer,
                population_size=population_size,
            ),
            static_argnames="num_steps",
        )

        self._random_key = jax.random.PRNGKey(self.seed)

    def make_initial_training_state(self) -> TrainingState:
        return make_initial_training_state(
            self._networks,
            self._critic_optimizer,
            self._twin_critic_optimizer,
            self._policy_optimizer,
            self._random_key,
            self.population_size,
        )

    def select_action(
        self,
        policy_params: PolicyParams,
        hyperparams: HyperParams,
        observation: Observation,
        exploration: bool = False,
    ) -> Action:
        action, self._random_key = self._select_action(
            policy_params, hyperparams, observation, self._random_key, exploration
        )
        return action

    def update_step(
        self,
        training_state: TrainingState,
        hyperparams: HyperParams,
        transition_batch: Transition,
        num_steps: int,
    ) -> TrainingState:
        return self._update_step(
            training_state, hyperparams, transition_batch, num_steps
        )

    def reset_optimizers(self, training_state: TrainingState) -> TrainingState:
        return training_state._replace(
            target_policy_params=training_state.policy_params,
            policy_opt_state=self.make_initial_training_state().policy_opt_state,
        )
