from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from fastpbrl.agents.abstract import Agent
from fastpbrl.sac.core import (
    make_default_networks,
    make_initial_training_state,
    select_action,
    update_step,
)
from fastpbrl.types import (
    Action,
    HyperParams,
    Observation,
    PolicyParams,
    TrainingState,
    Transition,
)
from fastpbrl.utils import make_gym_env


class SAC(Agent):
    def __init__(
        self,
        gym_env_name: str,
        seed: int,
        hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    ):
        self.seed = seed
        self.hidden_layer_sizes = hidden_layer_sizes

        environment = make_gym_env(gym_env_name)

        self._networks = make_default_networks(
            gym_env=environment, hidden_layer_sizes=self.hidden_layer_sizes
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

        self._update_step = jax.jit(
            partial(
                update_step,
                networks=self._networks,
                critic_optimizer=self._critic_optimizer,
                policy_optimizer=self._policy_optimizer,
                alpha_optimizer=self._alpha_optimizer,
            ),
            static_argnames="num_steps",
        )

        self._random_key = jax.random.PRNGKey(self.seed)

    def make_initial_training_state(
        self,
    ) -> TrainingState:
        return make_initial_training_state(
            networks=self._networks,
            critic_optimizer=self._critic_optimizer,
            policy_optimizer=self._policy_optimizer,
            alpha_optimizer=self._alpha_optimizer,
            random_key=self._random_key,
        )

    def select_action(
        self,
        policy_params: PolicyParams,
        hyperparams: HyperParams,
        observation: Observation,
        exploration: bool = False,
    ) -> Action:
        action, self._random_key = self._select_action(
            policy_params=policy_params,
            observation=observation,
            hyperparams=hyperparams,
            rng=self._random_key,
            exploration=exploration,
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
            state=training_state,
            hyperparams=hyperparams,
            transition_batch=transition_batch,
            num_steps=num_steps,
        )

    def reset_optimizers(self, training_state: TrainingState) -> TrainingState:
        initial_training_state = self.make_initial_training_state()
        return training_state._replace(
            policy_opt_state=initial_training_state.policy_opt_state,
            alpha_opt_state=initial_training_state.alpha_opt_state,
            alpha_params=initial_training_state.alpha_params,
        )
