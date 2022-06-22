from multiprocessing.managers import SharedMemoryManager
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from fastpbrl.types import NestedArray, NestedJaxArray, NestedNumpyArray


class NestedSharedMemory:
    """Object used to synchronize pytrees between processes."""

    def __init__(
        self, nested_array: NestedArray, shared_memory_manager: SharedMemoryManager
    ):
        """Initialization of the NestedSharedMemory object.

        Args:
            nested_array (NestedArray): Nested array that needs to be shared between
                processes, the structure will be used to initialize the shared_memory.
                Moreover, during the initialization of the object, the value of the
                nested_array will be set in at the shared memory location.
            shared_memory_manager (SharedMemoryManager): SharedMemoryManager from the
                multiprocessing.managers module. It is used to manage the memory slots
                used for storing the arrays.
        """

        # Create shared memory instances
        self._nested_shared_memory_space = jax.tree_map(
            lambda x: shared_memory_manager.SharedMemory(size=x.nbytes), nested_array
        )

        # Initialize the nested arrays
        self._original_nested_array = nested_array
        self._nested_shared_numpy_array = self._get_nested_shared_numpy_array()
        self.update(source_nested_array=nested_array)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        del state["_nested_shared_numpy_array"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._nested_shared_numpy_array = self._get_nested_shared_numpy_array()

    @property
    def original_nested_array(self) -> NestedArray:
        return self._original_nested_array

    @property
    def nested_shared_numpy_array(self) -> NestedNumpyArray:
        return self._nested_shared_numpy_array

    def retrieve(self) -> NestedJaxArray:
        """Create a NestedJaxArray containing the values stored in the shared memory.

        Returns:
            A NestedJaxArray containing the values stored in the shared memory.
        """
        return jax.tree_map(
            lambda np_array: jnp.array(np_array), self._nested_shared_numpy_array
        )

    def retrieve_with_index(self, index: int) -> NestedJaxArray:
        """Retrieve the subarrays of index 'index' from the pytree stored in the shared memory.

        Args:
            index: Index to retrieve.

        Returns:
            A NestedJaxArray subarray of the first 'num_rows' of the array stored in
                the shared memory.
        """
        return jax.tree_map(
            lambda np_array: jnp.array(np_array[index]), self._nested_shared_numpy_array
        )

    def retrieve_first_rows(self, num_rows: int) -> NestedJaxArray:
        """Retrieve the first 'num_rows' first rows from the pytree stored on the
            shared memory.This method is useful when retrieving the first N states in
            CEM-RL.

        Args:
            num_rows (int): Number of rows to retrieve.

        Returns:
            A NestedJaxArray subarray of the first 'num_rows' of the array stored in
                the shared memory.
        """
        return jax.tree_map(
            lambda np_array: jnp.array(np_array[:num_rows]),
            self._nested_shared_numpy_array,
        )

    def update(
        self,
        source_nested_array: NestedArray,
        is_subarray: bool = False,
    ) -> None:
        """num rows assume that the nested_array is a subpart of the nested_shared_numpy_array
        if it is specified,

        Args:
            source_nested_array (NestedArray): Nested array that will be used to update
                the shared_memory.
            is_subarray (bool): Boolean indicating weather the source_nested_array is a
                subarray of the source nested array. This is used when training an agent
                using CEM-RL in order to update the first N rows.
        """
        assert (
            jax.tree_util.tree_flatten(source_nested_array)[1]
            == jax.tree_util.tree_flatten(self._nested_shared_numpy_array)[1]
        )
        num_rows = (
            jax.tree_leaves(source_nested_array)[0].shape[0] if is_subarray else None
        )

        for jax_array, np_array in zip(
            jax.tree_util.tree_flatten(source_nested_array)[0],
            jax.tree_util.tree_flatten(self._nested_shared_numpy_array)[0],
        ):
            np_array[:num_rows] = np.array(
                jax.device_put(jax_array, jax.devices("cpu")[0]),
                copy=True,
            )

    def _get_nested_shared_numpy_array(self) -> NestedNumpyArray:
        return jax.tree_multimap(
            lambda array, shm: np.ndarray(
                array.shape, dtype=array.dtype, buffer=shm.buf
            ),
            self._original_nested_array,
            self._nested_shared_memory_space,
        )
