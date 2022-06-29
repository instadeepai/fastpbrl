"""
The code is in this file was adapted from https://github.com/deepmind/acme
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import haiku
import jax
import jax.numpy as jnp
import tensorflow_probability

hk_init = haiku.initializers
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
uniform_initializer = haiku.initializers.UniformScaling(scale=0.333)


@dataclass
class TanhToSpec:
    """Squashes real-valued inputs to match a BoundedArraySpec."""

    min_value: jnp.ndarray
    max_value: jnp.ndarray

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        scale = self.max_value - self.min_value
        offset = self.min_value
        inputs = jax.lax.tanh(inputs)  # [-1, 1]
        inputs = 0.5 * (inputs + 1.0)  # [0, 1]
        output = inputs * scale + offset  # [minimum, maximum]
        return output


class LayerNormMLP(haiku.Module):
    """Simple feedforward MLP torso with initial layer-norm.

    This module is an MLP which uses LayerNorm (with a tanh normalizer) on the
    first layer and non-linearities (elu) on all but the last remaining layers.
    """

    def __init__(self, layer_sizes: Sequence[int], activate_final: bool = False):
        """Construct the MLP.

        Args:
          layer_sizes: a sequence of ints specifying the size of each layer.
          activate_final: whether or not to use the activation function on the final
            layer of the neural network.
        """
        super().__init__(name="feedforward_mlp_torso")

        self._network = haiku.Sequential(
            [
                haiku.Linear(layer_sizes[0], w_init=uniform_initializer),
                haiku.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.lax.tanh,
                haiku.nets.MLP(
                    layer_sizes[1:],
                    w_init=uniform_initializer,
                    activation=jax.nn.elu,
                    activate_final=activate_final,
                ),
            ]
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Forwards the policy network."""
        return self._network(inputs)


class NearZeroInitializedLinear(haiku.Linear):
    """Simple linear layer, initialized at near zero weights and zero biases."""

    def __init__(self, output_size: int, scale: float = 1e-4):
        super().__init__(output_size, w_init=haiku.initializers.VarianceScaling(scale))


class TanhTransformedDistribution(tfd.TransformedDistribution):
    """Distribution followed by tanh."""

    def __init__(self, distribution, threshold=0.999, validate_args=False):
        """Initialize the distribution.

        Args:
          distribution: The distribution to transform.
          threshold: Clipping value of the action when computing the logprob.
          validate_args: Passed to super class.
        """
        super().__init__(
            distribution=distribution,
            bijector=tfp.bijectors.Tanh(),
            validate_args=validate_args,
        )
        # Computes the log of the average probability distribution outside the
        # clipping range, i.e. on the interval [-inf, -atanh(threshold)] for
        # log_prob_left and [atanh(threshold), inf] for log_prob_right.
        self._threshold = threshold
        inverse_threshold = self.bijector.inverse(threshold)
        # average(pdf) = p/epsilon
        # So log(average(pdf)) = log(p) - log(epsilon)
        log_epsilon = jnp.log(1.0 - threshold)
        # Those 2 values are differentiable w.r.t. model parameters, such that the
        # gradient is defined everywhere.
        self._log_prob_left = (
            self.distribution.log_cdf(-inverse_threshold) - log_epsilon
        )
        self._log_prob_right = (
            self.distribution.log_survival_function(inverse_threshold) - log_epsilon
        )

    def log_prob(self, event):
        # Without this clip there would be NaNs in the inner tf.where and that
        # causes issues for some reasons.
        event = jnp.clip(event, -self._threshold, self._threshold)
        # The inverse image of {threshold} is the interval [atanh(threshold), inf]
        # which has a probability of "log_prob_right" under the given distribution.
        return jnp.where(
            event <= -self._threshold,
            self._log_prob_left,
            jnp.where(
                event >= self._threshold, self._log_prob_right, super().log_prob(event)
            ),
        )

    def mode(self):
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


class NormalTanhDistribution(haiku.Module):
    """Module that produces a TanhTransformedDistribution distribution."""

    def __init__(
        self,
        num_dimensions: int,
        min_scale: float = 1e-3,
        w_init: hk_init.Initializer = hk_init.VarianceScaling(1.0, "fan_in", "uniform"),
        b_init: hk_init.Initializer = hk_init.Constant(0.0),
    ):
        """Initialization.

        Args:
          num_dimensions: Number of dimensions of a distribution.
          min_scale: Minimum standard deviation.
          w_init: Initialization for linear layer weights.
          b_init: Initialization for linear layer biases.
        """
        super().__init__(name="Normal")
        self._min_scale = min_scale
        self._loc_layer = haiku.Linear(num_dimensions, w_init=w_init, b_init=b_init)
        self._scale_layer = haiku.Linear(num_dimensions, w_init=w_init, b_init=b_init)

    def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
        loc = self._loc_layer(inputs)
        scale = self._scale_layer(inputs)
        scale = jax.nn.softplus(scale) + self._min_scale
        distribution = tfd.Normal(loc=loc, scale=scale)
        return tfd.Independent(
            TanhTransformedDistribution(distribution), reinterpreted_batch_ndims=1
        )


class AtariTorso(haiku.Module):
    """Simple convolutional stack commonly used for Atari."""

    def __init__(self):
        super().__init__(name="atari_torso")

        self._network = haiku.Sequential(
            [
                haiku.Conv2D(16, [8, 8], 4, padding="VALID"),
                jax.nn.relu,
                haiku.Conv2D(32, [4, 4], 2, padding="VALID"),
                jax.nn.relu,
                haiku.Conv2D(64, [3, 3], 1, padding="VALID"),
                jax.nn.relu,
            ]
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        inputs_rank = jnp.ndim(inputs)
        batched_inputs = inputs_rank == 4
        if inputs_rank < 3 or inputs_rank > 4:
            raise ValueError("Expected input BHWC or HWC. Got rank %d" % inputs_rank)

        outputs = self._network(inputs)

        if batched_inputs:
            return jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
        return jnp.reshape(outputs, [-1])  # [D]


class VectorizedAtariTorso(haiku.Module):
    """(manually) Vectorized version of AtariTorso for a population of networks"""

    def __init__(self, population_size: int):
        super().__init__(name="atari_torso")

        self._population_size = population_size
        self._network = haiku.Sequential(
            [
                haiku.Conv2D(
                    16 * population_size,
                    [8, 8],
                    4,
                    feature_group_count=population_size,
                    padding="VALID",
                ),
                jax.nn.relu,
                haiku.Conv2D(
                    32 * population_size,
                    [4, 4],
                    2,
                    feature_group_count=population_size,
                    padding="VALID",
                ),
                jax.nn.relu,
                haiku.Conv2D(
                    64 * population_size,
                    [3, 3],
                    1,
                    feature_group_count=population_size,
                    padding="VALID",
                ),
                jax.nn.relu,
            ]
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        inputs_rank = jnp.ndim(inputs)
        batched_inputs = inputs_rank == 4
        if inputs_rank < 3 or inputs_rank > 4:
            raise ValueError("Expected input BHWC or HWC. Got rank %d" % inputs_rank)

        outputs = self._network(inputs)

        if batched_inputs:
            return jnp.reshape(
                outputs, [self._population_size, outputs.shape[0], -1]
            )  # [B, D]
        return jnp.reshape(outputs, [self._population_size, -1])  # [D]


class DuellingMLP(haiku.Module):
    """A Duelling MLP Q-network."""

    def __init__(
        self,
        num_actions: int,
        hidden_sizes: Sequence[int],
        w_init: Optional[haiku.initializers.Initializer] = None,
    ):
        super().__init__(name="duelling_q_network")

        self._value_mlp = haiku.nets.MLP([*hidden_sizes, 1], w_init=w_init)
        self._advantage_mlp = haiku.nets.MLP(
            [*hidden_sizes, num_actions], w_init=w_init
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the duelling network.

        Args:
            inputs: 2-D tensor of shape [batch_size, embedding_size].

        Returns:
            q_values: 2-D tensor of action values of shape [batch_size, num_actions]
        """

        # Compute value & advantage for duelling.
        value = self._value_mlp(inputs)  # [B, 1]
        advantages = self._advantage_mlp(inputs)  # [B, A]

        # Advantages have zero mean.
        advantages -= jnp.mean(advantages, axis=-1, keepdims=True)  # [B, A]

        q_values = value + advantages  # [B, A]

        return q_values
