import logging
from dataclasses import dataclass
from typing import List, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from fastpbrl.types import NestedArray, NestedJaxArray


class CrossEntropyEntry(NamedTuple):
    parameters: NestedArray
    fitness: float


@dataclass
class CrossEntropyParameters:
    num_elites: int
    initial_diagonal_std: float = 1e-2
    final_cov_noise: float = 1e-5
    diagonal_tau_cov: float = 0.95
    use_biased_weights: bool = True


class CrossEntropyMethod:
    def __init__(
        self, cem_params: CrossEntropyParameters, initial_params: NestedJaxArray
    ):
        self._cem_params = cem_params
        self._cov_noise = self._cem_params.initial_diagonal_std

        self._entry_list: List[CrossEntropyEntry] = []
        self._params_mean: NestedArray = jax.tree_map(
            lambda x: x, initial_params
        )  # makes a copy of the params
        self._params_var: NestedArray = jax.tree_map(
            lambda x: jnp.ones_like(x) * self._cov_noise, initial_params
        )

        self._first_sampling = True

        if self._cem_params.use_biased_weights:
            self._elite_weights = np.log(
                (self._cem_params.num_elites + 1)
                / np.arange(1, self._cem_params.num_elites + 1)
            )
        else:
            self._elite_weights = np.ones(self._cem_params.num_elites)
        self._elite_weights /= self._elite_weights.sum()

        self._logger = logging.getLogger(f"{__name__}")

    def add(self, params: NestedJaxArray, fitness: float):
        if (
            len(self._entry_list) < self._cem_params.num_elites
            or self._entry_list[-1].fitness < fitness
        ):
            self._entry_list.append(
                CrossEntropyEntry(parameters=params, fitness=fitness)
            )
            self._entry_list = sorted(
                self._entry_list, key=lambda entry: entry.fitness, reverse=True
            )[: self._cem_params.num_elites]

    def get_mean(self) -> NestedJaxArray:
        return jax.tree_map(
            lambda x: x, self._params_mean
        )  # makes a copy of the parameters

    def sample(self, num_samples: int) -> List[NestedJaxArray]:
        if self._first_sampling:
            self._first_sampling = False
        else:
            self._update_mean_std()

        self._entry_list = []

        positive_noise = [
            jax.tree_map(
                lambda p: np.sqrt(p) * np.random.randn(*p.shape), self._params_var
            )
            for _ in range((num_samples + 1) // 2)
        ]
        negative_noise = [
            jax.tree_map(lambda p: -p, params)
            for params in positive_noise[: num_samples // 2]
        ]
        params_list = [
            jax.tree_map(
                lambda mean, noise: jnp.array(mean + noise),
                self._params_mean,
                noise_params,
            )
            for noise_params in positive_noise + negative_noise
        ]

        return params_list

    def _update_mean_std(self):
        assert len(self._entry_list) == self._cem_params.num_elites
        params_list = [e.parameters for e in self._entry_list]

        old_params_mean = jax.tree_map(lambda x: x, self._params_mean)  # make a copy

        # Update mean
        def weighted_mean(params_list: List[NestedArray]):
            return np.sum(
                [p * w for p, w in zip(params_list, self._elite_weights)], axis=0
            )

        self._params_mean = jax.tree_multimap(lambda *e: weighted_mean(e), *params_list)

        self._cov_noise = (
            self._cem_params.diagonal_tau_cov * self._cov_noise
            + (1 - self._cem_params.diagonal_tau_cov) * self._cem_params.final_cov_noise
        )

        # Update variance using the old mean as in Hansen (2016) Eq. 12
        def variance(mean_params, params_tuple):
            step_1 = np.power(jnp.array(params_tuple) - mean_params, 2)
            step_2 = np.sum(
                [p * w for p, w in zip(step_1, self._elite_weights)], axis=0
            )
            return step_2 / self._cem_params.num_elites + self._cov_noise

        self._params_var = jax.tree_multimap(
            lambda m, *p: variance(m, p),
            old_params_mean,
            *params_list,
        )
