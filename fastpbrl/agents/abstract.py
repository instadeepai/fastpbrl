from abc import ABC, abstractmethod
from typing import Callable, List

from fastpbrl.types import (
    Action,
    HyperParams,
    Observation,
    PolicyParams,
    PRNGKey,
    TrainingState,
    Transition,
)
from fastpbrl.utils import jax_tree_stack


class Agent(ABC):
    """Generic agent class common to TD3, SAC, CEM-RL, ..."""

    @abstractmethod
    def make_initial_training_state(self) -> TrainingState:
        """
        Generates initial values for the training state (weights for
            the policy neural network, weights for the crictic neural
            networks, ...)
        """

    @abstractmethod
    def select_action(
        self,
        policy_params: PolicyParams,
        hyperparams: HyperParams,
        observation: Observation,
        exploration: bool = False,
    ) -> Action:
        """
        Takes an observation as input and returns an action.

        Parameters:
            policy_params (PolicyParams): weights for the policy network to
                use to select the action.
            hyperparams (HyperParams): hyper parameters (such as amount of
                noise if exploration is true) to use to select the action.
            observation (Observation): observation from the gym environment.
            exploration (bool): whether to add exploration noise when selecting
                the action.

        Return:
            The action to use to step the gym environment.
        """

    @abstractmethod
    def update_step(
        self,
        training_state: TrainingState,
        hyperparams: HyperParams,
        transition_batch: Transition,
        num_steps: int,
    ) -> TrainingState:
        """
        Compute losses on a batch of transition data, derive the gradients
            with respect to the policy and critic parameters, step the optimizers
            and update the training state.

        Parameters:
            training_state (TrainingState): training state before the update
            hyperparams (HyperParams): hyperparameters (such as learning rates)
                to use to carry out the updates.
            transition_batch (Transition): batches of transitions to carry
                out num_steps update steps. For each leaf transition_batch:
                    - the first dimension should be equal to num_steps,
                    - the second dimension is the batch size.
            num_steps (int): Number of update steps to carry out at once.

        Return:
            The updated training state after carrying out num_steps update steps.
        """

    @abstractmethod
    def reset_optimizers(self, training_state: TrainingState) -> TrainingState:
        """
        Reset the optimizer states stored as part of training_state to their initial
            states. This is useful for instance in CEM-RL where new policy weights are
            sampled at each iteration and the optimizer states become invalid.

        Parameters:
            training_state (TrainingState): training state before the update

        Return:
            The updated training state after resetting the optimizer states.
        """


class PBTAgent(Agent):
    @abstractmethod
    def __init__(self, population_size: int, random_key: PRNGKey):
        self.population_size = population_size
        self._random_key = random_key

    @abstractmethod
    def reset_optimizers(
        self, training_state: TrainingState, index_list: List[int]
    ) -> TrainingState:
        pass

    def get_hyperparams_sampler_function(self) -> Callable[[], HyperParams]:
        return self._make_initial_hyperparams

    @abstractmethod
    def _make_initial_hyperparams(self) -> HyperParams:
        pass

    @abstractmethod
    def _make_initial_training_state(self) -> TrainingState:
        pass

    def make_initial_hyperparams(self) -> HyperParams:
        hyperparams = [
            self._make_initial_hyperparams() for _ in range(self.population_size)
        ]
        return jax_tree_stack(hyperparams)

    def make_initial_training_state(self) -> TrainingState:
        state_list = [
            self._make_initial_training_state() for _ in range(self.population_size)
        ]
        return jax_tree_stack(state_list)

    def _select_action(self, **kwargs):
        raise NotImplementedError

    def select_action(
        self,
        policy_params: PolicyParams,
        hyperparams: HyperParams,
        observation: Observation,
        exploration: bool = False,
    ) -> Action:
        action, self._random_key = self._select_action(
            policy_params=policy_params,
            hyperparams=hyperparams,
            observation=observation,
            rng=self._random_key,
            exploration=exploration,
        )
        return action

    def _update_step(self, *args):
        raise NotImplementedError

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
