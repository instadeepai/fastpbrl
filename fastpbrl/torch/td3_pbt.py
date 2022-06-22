import torch
import tree

from fastpbrl.td3.core import TD3HyperParams
from fastpbrl.torch.networks import (
    VectorizedContinuousDeterministicActor,
    VectorizedContinuousQNetwork,
    soft_update_from_to,
)
from fastpbrl.torch.td3 import td3_critic_loss
from fastpbrl.types import Transition


class TD3PBT:
    """Implements the Twin Delayed Deep Deterministic Policy Gradient (TD3) agent
    for a population of agents.
    """

    def __init__(
        self,
        policy_network: VectorizedContinuousDeterministicActor,
        critic_network: VectorizedContinuousQNetwork,
        target_policy_network: VectorizedContinuousDeterministicActor,
        target_critic_network: VectorizedContinuousQNetwork,
        hyperparams: TD3HyperParams,
        min_action: torch.Tensor,
        max_action: torch.Tensor,
        population_size: int,
    ):
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._target_policy_network = target_policy_network
        self._target_critic_network = target_critic_network
        self._hyperparams = hyperparams
        self._min_action = min_action
        self._max_action = max_action
        self._population_size = population_size

        self._target_policy_network.load_state_dict(self._policy_network.state_dict())
        self._target_critic_network.load_state_dict(self._critic_network.state_dict())

        self._policy_optimizer = torch.optim.Adam(
            self._policy_network.parameters(), lr=self._hyperparams.lr_policy
        )

        self._critic_optimizer = torch.optim.Adam(
            self._critic_network.parameters(), lr=self._hyperparams.lr_critic
        )

        for parameter in self._target_policy_network.parameters():
            parameter.requires_grad = False

        for parameter in self._target_critic_network.parameters():
            parameter.requires_grad = False

        self._n_step_since_policy_update = 0

    def train_step(self, transition_batch: Transition, num_steps: int):
        assert transition_batch.reward.shape[0:2] == (num_steps, self._population_size)

        for step_id in range(num_steps):
            self._single_train_step(
                tree.map_structure(
                    lambda x: x[step_id],
                    transition_batch,
                )
            )

    def _single_train_step(self, transition_batch: Transition):
        critic_loss = td3_critic_loss(
            transition_batch,
            target_policy_network=self._target_policy_network,
            critic_network=self._critic_network,
            target_critic_network=self._target_critic_network,
            hyperparams=self._hyperparams,
            min_action=self._min_action,
            max_action=self._max_action,
        )

        # Rescale critic loss to sum (as opposed to average) critic losses
        # over the population
        critic_loss *= self._population_size

        # Update critic parameters
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        self._n_step_since_policy_update += 1
        if self._n_step_since_policy_update % self._hyperparams.delay == 0:
            # Compute policy loss
            policy_loss = (
                -self._critic_network.forward_q1(
                    transition_batch.observation,
                    self._policy_network(transition_batch.observation),
                )
                .sum(dim=0)
                .mean()
            )

            # Update policy parameters
            self._policy_optimizer.zero_grad()
            policy_loss.backward()
            self._policy_optimizer.step()

            # Update target networks
            soft_update_from_to(
                self._critic_network,
                self._target_critic_network,
                self._hyperparams.tau,
            )

            soft_update_from_to(
                self._policy_network,
                self._target_policy_network,
                self._hyperparams.tau,
            )
