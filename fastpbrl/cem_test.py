import jax
import jax.numpy as jnp
import numpy as np

from fastpbrl.cem import CrossEntropyMethod, CrossEntropyParameters
from fastpbrl.types import NestedJaxArray


def test_cem():
    """Test CEM for a simple optimization problem with four variables w, x, y, z
    where the objective is to minimize |w - 1| + |x - 1| + |y - 1| + |z - 1|.
    """
    np.random.seed(0)

    def check_consistency(
        nested_array_a: NestedJaxArray, nested_array_b: NestedJaxArray
    ) -> bool:
        return (
            jax.tree_util.tree_flatten(nested_array_a)[1]
            == jax.tree_util.tree_flatten(nested_array_b)[1]
        )

    nb_iter = 100
    population_size = 100
    num_elites = population_size // 2

    initial_params = [jnp.array(np.random.rand(1, 2)), jnp.array(np.random.rand(2, 1))]
    cem_params = CrossEntropyParameters(
        num_elites=num_elites,
        initial_diagonal_std=1.0,
        final_cov_noise=0.001,
        diagonal_tau_cov=0.95,
    )
    cem = CrossEntropyMethod(cem_params, initial_params=initial_params)

    for _ in range(nb_iter):
        all_sampled_params = cem.sample(num_samples=population_size)
        assert len(all_sampled_params) == population_size
        assert check_consistency(initial_params, all_sampled_params[0])

        for params in all_sampled_params:
            fitness = -sum(
                [
                    np.sum(np.abs(np.ones_like(np_array) - np_array))
                    for np_array in params
                ]
            )
            cem.add(params, fitness=fitness)

        assert check_consistency(initial_params, cem.get_mean())

    for np_array in cem.get_mean():
        np.testing.assert_allclose(
            np_array, np.ones_like(np_array), atol=np.sqrt(cem_params.final_cov_noise)
        )
