import torch
from torch.nn.functional import mse_loss

from fastpbrl.dqn.core import DQNHyperParameters
from fastpbrl.torch.networks import VectorizedDQNAtariCritic, soft_update_from_to
from fastpbrl.types import Transition


class DQNPBT:
    def __init__(
        self,
        critic_network: VectorizedDQNAtariCritic,
        target_critic_network: VectorizedDQNAtariCritic,
        hyperparams: DQNHyperParameters,
        population_size: int,
    ):

        self._critic_network = critic_network
        self._target_critic_network = target_critic_network
        self._hyperparams = hyperparams
        self._population_size = population_size

        self._critic_optimizer = torch.optim.Adam(
            params=self._critic_network.parameters(),
            lr=self._hyperparams.learning_rate,
        )

        self._target_critic_network.load_state_dict(self._critic_network.state_dict())

        for parameter in self._target_critic_network.parameters():
            parameter.requires_grad = False

        self._n_step_since_target_update = 0

    def train_step(self, transition_batch: Transition, num_steps: int):
        assert transition_batch.reward.shape[1] % num_steps == 0
        batch_size = transition_batch.reward.shape[1] // num_steps

        assert transition_batch.action.shape == (
            self._population_size,
            batch_size * num_steps,
        )
        assert transition_batch.done.shape == (
            self._population_size,
            batch_size * num_steps,
        )
        assert transition_batch.reward.shape == (
            self._population_size,
            batch_size * num_steps,
        )
        assert transition_batch.observation.shape[0] == num_steps * batch_size

        for step_id in range(num_steps):
            self._single_train_step(
                Transition(
                    action=transition_batch.action[
                        :, step_id * batch_size : (step_id + 1) * batch_size
                    ],
                    done=transition_batch.done[
                        :, step_id * batch_size : (step_id + 1) * batch_size
                    ],
                    reward=transition_batch.reward[
                        :, step_id * batch_size : (step_id + 1) * batch_size
                    ],
                    observation=transition_batch.observation[
                        step_id * batch_size : (step_id + 1) * batch_size
                    ],
                    next_observation=transition_batch.next_observation[
                        step_id * batch_size : (step_id + 1) * batch_size
                    ],
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
        total_loss = 0.0
        for i in range(self._population_size):
            total_loss += mse_loss(
                target_q_tm1[i],
                q_tm1[i, torch.arange(q_tm1.size(1)), transition_batch.action[i]],
            )

        self._critic_optimizer.zero_grad()
        total_loss.backward()
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
