"""
The code is in this file was adapted from https://github.com/DLR-RM/stable-baselines3
"""

import torch
import tree
from torch.nn.functional import mse_loss

from fastpbrl.td3.core import TD3HyperParams
from fastpbrl.torch.networks import (
    ContinuousDeterministicActor,
    ContinuousQNetwork,
    soft_update_from_to,
)
from fastpbrl.types import Transition


def td3_critic_loss(
    transition_batch: Transition,
    target_policy_network: torch.nn.Module,
    critic_network: torch.nn.Module,
    target_critic_network: torch.nn.Module,
    hyperparams: TD3HyperParams,
    min_action: torch.Tensor,
    max_action: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        # Select action according to policy and add clipped noise
        noise = hyperparams.target_sigma * torch.randn_like(transition_batch.action)
        noise = noise.clamp(-hyperparams.noise_clip, hyperparams.noise_clip)

        next_action = target_policy_network(transition_batch.next_observation) + noise
        next_action = next_action.clamp(min_action, max_action)

        # Compute the target Q value
        target_q1, target_q2 = target_critic_network(
            transition_batch.next_observation, next_action
        )
        min_q = torch.min(target_q1, target_q2)
        target_q = (
            1.0 - transition_batch.done
        ) * hyperparams.discount * min_q + transition_batch.reward

    # Get current Q estimates
    current_q1, current_q2 = critic_network(
        transition_batch.observation, transition_batch.action
    )

    # Return critic loss
    return mse_loss(current_q1, target_q) + mse_loss(current_q2, target_q)


class TD3:
    """Implements the Twin Delayed Deep Deterministic Policy Gradient (TD3) agent."""

    def __init__(
        self,
        policy_network: ContinuousDeterministicActor,
        critic_network: ContinuousQNetwork,
        target_policy_network: ContinuousDeterministicActor,
        target_critic_network: ContinuousQNetwork,
        hyperparams: TD3HyperParams,
        min_action: torch.Tensor,
        max_action: torch.Tensor,
    ):
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._target_policy_network = target_policy_network
        self._target_critic_network = target_critic_network
        self._hyperparams = hyperparams
        self._min_action = min_action
        self._max_action = max_action

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

        self._num_steps_since_policy_update = 0

    def train_step(self, transition_batch: Transition, num_steps: int):
        assert transition_batch.reward.shape[0] == num_steps

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

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        self._num_steps_since_policy_update += 1
        if self._num_steps_since_policy_update % self._hyperparams.delay == 0:
            # Compute policy loss if needed
            policy_loss = -self._critic_network.forward_q1(
                transition_batch.observation,
                self._policy_network(transition_batch.observation),
            ).mean()

            self._policy_optimizer.zero_grad()
            policy_loss.backward()
            self._policy_optimizer.step()

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
