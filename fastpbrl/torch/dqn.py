"""
The code is in this file was adapted from https://github.com/DLR-RM/stable-baselines3
"""

import torch
import tree
from torch.nn.functional import mse_loss

from fastpbrl.dqn.core import DQNHyperParameters
from fastpbrl.torch.networks import DQNAtariCritic, soft_update_from_to
from fastpbrl.types import Transition


class DQN:
    def __init__(
        self,
        critic_network: DQNAtariCritic,
        target_critic_network: DQNAtariCritic,
        hyperparams: DQNHyperParameters,
    ):

        self._critic_network = critic_network
        self._target_critic_network = target_critic_network
        self._hyperparams = hyperparams

        self._critic_optimizer = torch.optim.Adam(
            params=self._critic_network.parameters(),
            lr=self._hyperparams.learning_rate,
        )

        self._target_critic_network.load_state_dict(self._critic_network.state_dict())

        for parameter in self._target_critic_network.parameters():
            parameter.requires_grad = False

        self._n_step_since_target_update = 0

    def train_step(self, transition_batch: Transition, num_steps: int):
        assert transition_batch.reward.shape[0] % num_steps == 0
        batch_size = transition_batch.reward.shape[0] // num_steps

        assert all(
            tree.flatten(
                tree.map_structure(
                    lambda x: x.shape[0] == batch_size * num_steps, transition_batch
                )
            )
        )

        for step_id in range(num_steps):
            self._single_train_step(
                tree.map_structure(
                    lambda x: x[step_id * batch_size : (step_id + 1) * batch_size],
                    transition_batch,
                )
            )

    def _single_train_step(self, transition_batch: Transition):
        with torch.no_grad():
            q_t = self._target_critic_network(transition_batch.next_observation)

            target_q_tm1 = (
                transition_batch.reward
                + self._hyperparams.discount
                * (1.0 - transition_batch.done)
                * torch.max(q_t, dim=-1)[0]
            )

        q_tm1 = self._critic_network(transition_batch.observation)
        q_values = q_tm1[torch.arange(q_tm1.size(0)), transition_batch.action]

        critic_loss = mse_loss(target_q_tm1, q_values)

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        self._n_step_since_target_update += 1
        if (
            self._n_step_since_target_update % self._hyperparams.target_update_period
            == 0
        ):
            soft_update_from_to(
                self._critic_network,
                self._target_critic_network,
                1.0,
            )
