import torch
import tree

from fastpbrl.sac.core import SACHyperParams
from fastpbrl.torch.networks import (
    VectorizedContinuousQNetwork,
    VectorizedSquashedGaussianActor,
    soft_update_from_to,
)
from fastpbrl.torch.sac import sac_critic_loss
from fastpbrl.types import Transition


class SACPBT:
    """Implements the Soft Actor Critic (SAC) agent for a population
    of agents.
    """

    def __init__(
        self,
        policy_network: VectorizedSquashedGaussianActor,
        critic_network: VectorizedContinuousQNetwork,
        target_critic_network: VectorizedContinuousQNetwork,
        hyperparams: SACHyperParams,
        population_size: int,
    ):

        self._policy_network = policy_network
        self._critic_network = critic_network
        self._target_critic_network = target_critic_network
        self._hyperparams = hyperparams
        self._population_size = population_size

        self._policy_optimizer = torch.optim.Adam(
            params=self._policy_network.parameters(),
            lr=self._hyperparams.lr_policy,
        )

        self._critic_optimizer = torch.optim.Adam(
            params=self._critic_network.parameters(),
            lr=self._hyperparams.lr_critic,
        )

        self._target_critic_network.load_state_dict(self._critic_network.state_dict())

        for parameter in self._target_critic_network.parameters():
            parameter.requires_grad = False

        if self._hyperparams.adaptive_entropy_coefficient:
            self._target_entropy = torch.Tensor([self._hyperparams.target_entropy])

            # load alpha on the same device as the networks
            device = next(policy_network.parameters()).device
            self._log_alpha = torch.zeros(1, requires_grad=True, device=device)

            # load target_entropy on the same device as the networks
            self._target_entropy = self._target_entropy.to(device)

            self._alpha_optimizer = torch.optim.Adam(
                params=[self._log_alpha], lr=self._hyperparams.lr_alpha
            )

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
        new_obs_actions, log_pi = self._policy_network(transition_batch.observation)

        # Compute alpha loss
        if self._hyperparams.adaptive_entropy_coefficient:
            alpha = self._log_alpha.exp()
            alpha_loss = (
                -(self._log_alpha * (log_pi + self._target_entropy).detach())
                .sum(dim=0)
                .mean()
            )
        else:
            alpha = self._hyperparams.entropy_coefficient

        # Compute policy losses
        q1_new_actions, q2_new_actions = self._target_critic_network(
            transition_batch.observation, new_obs_actions
        )
        q_new_actions = torch.min(q1_new_actions, q2_new_actions)
        policy_loss = (alpha * log_pi - q_new_actions).sum(dim=0).mean()

        # Compute critic losses
        critic_loss = sac_critic_loss(
            transition_batch,
            policy_network=self._policy_network,
            critic_network=self._critic_network,
            target_critic_network=self._target_critic_network,
            alpha=alpha,
            hyperparams=self._hyperparams,
        )
        # Rescale critic loss to sum (as opposed to average) critic losses
        # over the population
        critic_loss *= self._population_size

        # Compute gradients and step optimizers
        if self._hyperparams.adaptive_entropy_coefficient:
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self._alpha_optimizer.step()

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        soft_update_from_to(
            self._critic_network,
            self._target_critic_network,
            self._hyperparams.tau,
        )
