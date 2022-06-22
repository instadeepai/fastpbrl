import math
from typing import List, Tuple

import numpy as np
import torch
import tree


def soft_update_from_to(source: torch.nn.Module, target: torch.nn.Module, tau: float):
    """
    Soft update parameters from target towards parameters from source such as
    theta_t <-- (1 - tau) * theta_t + tau * theta_s
    This function is mainly used to update target networks for TD updates.
    """
    with torch.no_grad():
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1 - tau)
            target_param.data.add_(tau * param.data)


def torch_load_on_device(
    tree_of_array: tree.StructureKV[str, np.ndarray], device: torch.device
) -> tree.StructureKV[str, torch.Tensor]:
    """Convert the leaves of a tree-like object from numpy arrays
    into torch tensors loaded on the specified device.
    """
    return tree.map_structure(
        lambda np_array: torch.from_numpy(np_array).to(device),
        tree_of_array,
    )


class MLP(torch.nn.Module):
    ACTIVATIONS = {
        "ReLU": torch.nn.ReLU,
        "SiLU": torch.nn.SiLU,
        "Tanh": torch.nn.Tanh,
        "LeakyReLU": torch.nn.LeakyReLU,
        "ELU": torch.nn.ELU,
    }

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers_dim: Tuple[int],
        activation="ReLU",
        use_layer_norm: bool = False,
        active_final: bool = False,
    ):
        torch.nn.Module.__init__(self)
        num_neurons = [input_dim] + list(layers_dim) + [output_dim]
        num_neurons = zip(num_neurons[:-1], num_neurons[1:])

        all_layers = []
        for layer_id, (in_dim, out_dim) in enumerate(num_neurons):
            all_layers.append(torch.nn.Linear(in_dim, out_dim))
            if use_layer_norm and layer_id == 0:
                all_layers.append(torch.nn.LayerNorm(out_dim))
            all_layers.append(MLP.ACTIVATIONS[activation]())

        if not active_final:
            all_layers.pop()  # remove last activation

        self._mlp = torch.nn.Sequential(*all_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._mlp(x)


class ContinuousQNetwork(torch.nn.Module):
    """
    Continuous State-Action Q function.
    Actually implements two independent Q-functions.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        layers_dim: Tuple[int],
        activation: str = "ReLU",
        use_layer_norm: bool = False,
    ):
        super().__init__()

        # q1 architecture
        self._q1_mlp = MLP(
            input_dim=observation_dim + action_dim,
            output_dim=1,
            layers_dim=layers_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )

        # q2 architecture
        self._q2_mlp = MLP(
            input_dim=observation_dim + action_dim,
            output_dim=1,
            layers_dim=layers_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass in both Q-value networks. Takes an observation and
        action, and returns the estimated value of taking this particular action in
        this state.
        """
        cat_obs_action = torch.cat([observation, action], -1)
        return self._q1_mlp(cat_obs_action), self._q2_mlp(cat_obs_action)

    def forward_q1(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cat_obs_action = torch.cat([observation, action], -1)
        return self._q1_mlp(cat_obs_action)


class ContinuousDeterministicActor(torch.nn.Module):
    """
    Continuous deterministic actor.
    Maps observations to continuous actions.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        layers_dim: Tuple[int],
        max_action: torch.Tensor,
        min_action: torch.Tensor,
        activation: str = "ReLU",
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self._core_mlp = MLP(
            observation_dim,
            action_dim,
            layers_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )
        self._max_action = max_action
        self._min_action = min_action

    def forward(self, observation) -> torch.Tensor:
        return (self._max_action - self._min_action) * 0.5 * (
            torch.tanh(self._core_mlp(observation)) + 1.0
        ) + self._min_action


class SquashedGaussianActor(torch.nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        layers_dim: List[int],
        max_action: torch.Tensor,
        min_action: torch.Tensor,
        activation: str = "ReLU",
        use_layer_norm: bool = False,
        min_std: float = 1e-3,
    ):
        super().__init__()

        assert len(layers_dim) > 1

        # define shared MLP network
        self._shared_mlp = MLP(
            input_dim=observation_dim,
            output_dim=layers_dim[-1],
            layers_dim=layers_dim[:-1],
            use_layer_norm=use_layer_norm,
            activation=activation,
            active_final=True,
        )

        # define mean and log-std linear layers for tanh normal distribution
        self._mean_layer = torch.nn.Linear(layers_dim[-1], action_dim)
        self._std_layer = torch.nn.Linear(layers_dim[-1], action_dim)

        self._max_action = max_action
        self._min_action = min_action
        self._min_std = min_std

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._shared_mlp(observation)
        mean, std = self._mean_layer(x), self._std_layer(x)

        base_gaussian = torch.distributions.Normal(
            mean, torch.nn.Softplus()(std) + self._min_std
        )
        tanh_normal = torch.distributions.TransformedDistribution(
            base_gaussian, [torch.distributions.TanhTransform(cache_size=1)]
        )

        action = tanh_normal.rsample()
        log_prob = tanh_normal.log_prob(action)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        action = (self._max_action - self._min_action) * (
            action + 1.0
        ) + self._min_action
        return action, log_prob


class VectorizedLinearLayer(torch.nn.Module):
    """Vectorized version of torch.nn.Linear."""

    def __init__(
        self,
        population_size: int,
        in_features: int,
        out_features: int,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self._population_size = population_size
        self._in_features = in_features
        self._out_features = out_features

        self.weight = torch.nn.Parameter(
            torch.empty(self._population_size, self._in_features, self._out_features),
            requires_grad=True,
        )
        self.bias = torch.nn.Parameter(
            torch.empty(self._population_size, 1, self._out_features),
            requires_grad=True,
        )

        for member_id in range(population_size):
            torch.nn.init.kaiming_uniform_(self.weight[member_id], a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)

        self._layer_norm = (
            torch.nn.LayerNorm(self._out_features, self._population_size)
            if use_layer_norm
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == self._population_size
        if self._layer_norm is not None:
            return self._layer_norm(x.matmul(self.weight) + self.bias)
        return x.matmul(self.weight) + self.bias


class VectorizedMLP(torch.nn.Module):
    """Vectorized version of MLP."""

    def __init__(
        self,
        population_size: int,
        input_dim: int,
        output_dim: int,
        layers_dim: Tuple[int],
        activation="ReLU",
        use_layer_norm: bool = False,
        active_final: bool = False,
    ):
        super().__init__()

        num_neurons = [input_dim] + list(layers_dim) + [output_dim]
        num_neurons = zip(num_neurons[:-1], num_neurons[1:])

        all_layers = []
        for layer_id, (in_dim, out_dim) in enumerate(num_neurons):
            all_layers.append(
                VectorizedLinearLayer(
                    population_size,
                    in_dim,
                    out_dim,
                    use_layer_norm=use_layer_norm and layer_id == 0,
                )
            )
            all_layers.append(MLP.ACTIVATIONS[activation]())

        if not active_final:
            all_layers.pop()  # remove last activation

        self._mlp = torch.nn.Sequential(*all_layers)
        self._population_size = population_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == self._population_size
        return self._mlp(x)


class VectorizedContinuousQNetwork(torch.nn.Module):
    """Vectorized version of ContinuousQNetwork."""

    def __init__(
        self,
        population_size: int,
        observation_dim: int,
        action_dim: int,
        layers_dim: Tuple[int],
        activation: str = "ReLU",
        use_layer_norm: bool = False,
    ):
        super().__init__()

        def _create_vectorized_mlp():
            return VectorizedMLP(
                population_size=population_size,
                input_dim=observation_dim + action_dim,
                output_dim=1,
                layers_dim=layers_dim,
                activation=activation,
                use_layer_norm=use_layer_norm,
            )

        self._q1_mlp = _create_vectorized_mlp()
        self._q2_mlp = _create_vectorized_mlp()
        self._population_size = population_size

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass in both Q-value networks. Takes an observation and
        action, and returns the estimated value of taking this particular action in
        this state.
        """
        assert observation.shape[0] == self._population_size
        assert action.shape[0] == self._population_size
        cat_obs_action = torch.cat([observation, action], -1)
        return self._q1_mlp(cat_obs_action), self._q2_mlp(cat_obs_action)

    def forward_q1(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert observation.shape[0] == self._population_size
        assert action.shape[0] == self._population_size
        cat_obs_action = torch.cat([observation, action], -1)
        return self._q1_mlp(cat_obs_action)


class VectorizedContinuousDeterministicActor(torch.nn.Module):
    """
    Vectorized implementation of ContinuousDeterministicActor for a population
    of ContinuousDeterministicActor neural networks
    """

    def __init__(
        self,
        population_size: int,
        observation_dim: int,
        action_dim: int,
        layers_dim: List[int],
        max_action: torch.Tensor,
        min_action: torch.Tensor,
        activation: str = "ReLU",
        use_layer_norm: bool = False,
    ):
        super().__init__()

        assert len(layers_dim) > 1

        self._core_mlp = VectorizedMLP(
            population_size,
            observation_dim,
            action_dim,
            layers_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )
        self._max_action = torch.repeat_interleave(
            torch.unsqueeze(max_action, dim=0), population_size, dim=0
        )
        self._min_action = torch.repeat_interleave(
            torch.unsqueeze(min_action, dim=0), population_size, dim=0
        )
        self._population_size = population_size

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        assert observation.shape[0] == self._population_size

        return (self._max_action - self._min_action)[:, None, :] * 0.5 * (
            torch.tanh(self._core_mlp(observation)) + 1.0
        ) + self._min_action[:, None, :]


class VectorizedSquashedGaussianActor(torch.nn.Module):
    def __init__(
        self,
        population_size: int,
        observation_dim: int,
        action_dim: int,
        layers_dim: List[int],
        max_action: torch.Tensor,
        min_action: torch.Tensor,
        activation: str = "ReLU",
        use_layer_norm: bool = False,
        min_std: float = 1e-3,
    ):
        super().__init__()

        assert len(layers_dim) > 1

        # define shared MLP network
        self._shared_mlp = VectorizedMLP(
            population_size=population_size,
            input_dim=observation_dim,
            output_dim=layers_dim[-1],
            layers_dim=layers_dim[:-1],
            use_layer_norm=use_layer_norm,
            activation=activation,
            active_final=True,
        )

        # define mean and log-std linear layers for tanh normal distribution
        self._mean_layer = VectorizedLinearLayer(
            population_size, layers_dim[-1], action_dim
        )
        self._std_layer = VectorizedLinearLayer(
            population_size, layers_dim[-1], action_dim
        )

        self._population_size = population_size
        self._max_action = torch.repeat_interleave(
            torch.unsqueeze(max_action, dim=0), population_size, dim=0
        )
        self._min_action = torch.repeat_interleave(
            torch.unsqueeze(min_action, dim=0), population_size, dim=0
        )
        self._min_std = min_std

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert observation.shape[0] == self._population_size

        # forward pass
        x = self._shared_mlp(observation)
        mean, std = self._mean_layer(x), self._std_layer(x)

        base_gaussian = torch.distributions.Normal(
            mean, torch.nn.Softplus()(std) + self._min_std
        )
        tanh_normal = torch.distributions.TransformedDistribution(
            base_gaussian, [torch.distributions.TanhTransform(cache_size=1)]
        )

        action = tanh_normal.rsample()
        log_prob = tanh_normal.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        action = (self._max_action - self._min_action)[:, None, :] * (
            action + 1.0
        ) + self._min_action[:, None, :]
        return action, log_prob


class DQNAtariCritic(torch.nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int],
        num_channels: int,
        num_actions: int,
        hidden_layer_sizes: Tuple[int, ...] = (256,),
    ):
        super().__init__()
        self._conv_network = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, 16, 8, 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 4, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
        )

        # Only a specific image format is supported for now
        assert image_shape == (84, 84)
        assert num_channels == 4
        input_dim = 3136

        self._value_mlp = MLP(
            input_dim,
            output_dim=1,
            layers_dim=hidden_layer_sizes,
        )

        self._advantage_mlp = MLP(
            input_dim,
            output_dim=num_actions,
            layers_dim=hidden_layer_sizes,
        )

    def forward(
        self,
        observation,
    ):
        obs_post_conv = self._conv_network(observation)
        obs_post_conv = torch.reshape(obs_post_conv, (obs_post_conv.shape[0], -1))

        # Compute value & advantage for duelling.
        value = self._value_mlp(obs_post_conv)
        advantages = self._advantage_mlp(obs_post_conv)

        # Advantages have zero mean.
        advantages -= torch.mean(advantages, dim=-1, keepdims=True)

        q_values = value + advantages
        return q_values


class VectorizedDQNAtariCritic(torch.nn.Module):
    def __init__(
        self,
        population_size: int,
        image_shape: Tuple[int],
        num_channels: int,
        num_actions: int,
        hidden_layer_sizes: Tuple[int, ...] = (256,),
    ):
        super().__init__()
        self._population_size = population_size

        self._conv_network = torch.nn.Sequential(
            *[
                torch.nn.Conv2d(
                    num_channels * population_size,
                    16 * population_size,
                    8,
                    4,
                    groups=population_size,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    16 * population_size,
                    32 * population_size,
                    4,
                    2,
                    groups=population_size,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    32 * population_size,
                    64 * population_size,
                    3,
                    1,
                    groups=population_size,
                ),
                torch.nn.ReLU(),
            ]
        )

        # Only a specific image format is supported for now
        assert image_shape == (84, 84)
        assert num_channels == 4
        input_dim = 3136

        self._value_mlp = VectorizedMLP(
            population_size=population_size,
            input_dim=input_dim,
            output_dim=1,
            layers_dim=hidden_layer_sizes,
        )

        self._advantage_mlp = VectorizedMLP(
            population_size=population_size,
            input_dim=input_dim,
            output_dim=num_actions,
            layers_dim=hidden_layer_sizes,
        )

    def forward(
        self,
        observation,
    ):
        obs_post_conv = self._conv_network(observation)
        obs_post_conv = torch.reshape(
            obs_post_conv, (self._population_size, obs_post_conv.shape[0], -1)
        )

        # Compute value & advantage for duelling.
        value = self._value_mlp(obs_post_conv)
        advantages = self._advantage_mlp(obs_post_conv)

        # Advantages have zero mean.
        advantages -= torch.mean(advantages, dim=-1, keepdims=True)

        q_values = value + advantages

        return q_values
