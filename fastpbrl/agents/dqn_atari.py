from functools import partial
from typing import Tuple

import jax
import optax

from fastpbrl.agents.abstract import Agent
from fastpbrl.dqn.core import (
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


class DQNAtari(Agent):
    def __init__(
        self,
        num_frame_stack: int,
        image_shape: Tuple[int],
        num_actions: int,
        seed: int,
        hidden_layer_sizes: Tuple[int, ...] = (256,),
    ):
        self._networks = make_default_networks(
            num_frame_stack=num_frame_stack,
            image_shape=image_shape,
            num_actions=num_actions,
            hidden_layer_sizes=hidden_layer_sizes,
        )

        self._select_action = jax.jit(
            partial(
                select_action,
                networks=self._networks,
            )
        )

        self._critic_optimizer = optax.adam(learning_rate=1.0)

        self._update_step = jax.jit(
            partial(
                update_step,
                networks=self._networks,
                critic_optimizer=self._critic_optimizer,
            ),
            static_argnames="num_steps",
        )

        self._random_key = jax.random.PRNGKey(seed)

    def make_initial_training_state(
        self,
    ) -> TrainingState:
        return make_initial_training_state(
            networks=self._networks,
            critic_optimizer=self._critic_optimizer,
            random_key=self._random_key,
        )

    def select_action(
        self,
        critic_params: PolicyParams,
        hyperparams: HyperParams,
        observation: Observation,
        exploration: bool = False,
    ) -> Action:
        return self._select_action(
            critic_params=critic_params,
            observation=observation,
            hyperparams=hyperparams,
        )

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
            critic_opt_state=initial_training_state.critic_opt_state,
            steps=0,
        )
