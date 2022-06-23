import multiprocessing
from multiprocessing.managers import SharedMemoryManager

import tree
import numpy as np
from typing import Dict, Any

from fastpbrl.types import NestedNumpyArray
from fastpbrl.utils import _update


class TreeQueue:
    """
    Meant as a replacement to multiprocessing.Queue when objects
    that will be stored in the queue are trees of numpy arrays of
    fixed shape and dtype.

    Elements of the queue are stored in shared memory which is pre-allocated
    at instantiation.
    """

    def __init__(
        self,
        nested_array: NestedNumpyArray,
        maxsize: int,
        shared_memory_manager: SharedMemoryManager,
    ):
        assert maxsize > 1
        self._maxsize = maxsize
        self._index_read = multiprocessing.Value("i", 0)
        self._index_write = multiprocessing.Value("i", 0)

        self._index_ready_for_read = [multiprocessing.Event() for _ in range(maxsize)]
        self._index_ready_for_write = [multiprocessing.Event() for _ in range(maxsize)]

        for event in self._index_ready_for_read:
            event.clear()
        for event in self._index_ready_for_write:
            event.set()

        # Create shared memory for the entire queue
        self._original_nested_array = tree.map_structure(
            lambda np_array: np.stack([np_array for _ in range(maxsize)]), nested_array
        )
        self._nested_shared_memory_space = tree.map_structure(
            lambda np_array: shared_memory_manager.SharedMemory(size=np_array.nbytes),
            self._original_nested_array,
        )
        self._nested_shared_numpy_array = self._get_nested_shared_numpy_array()

    def put(self, nested_array: NestedNumpyArray):
        with self._index_write.get_lock():
            index_write = self._index_write.value
            self._index_write.value = (index_write + 1) % self._maxsize
            self._index_ready_for_write[index_write].wait()
            self._index_ready_for_write[index_write].clear()

        self._nested_shared_numpy_array = tree.map_structure(
            lambda x, y: _update(x, y, index_write),
            nested_array,
            self._nested_shared_numpy_array,
        )
        self._index_ready_for_read[index_write].set()

    def get(self) -> NestedNumpyArray:
        with self._index_read.get_lock():
            index_read = self._index_read.value
            self._index_read.value = (index_read + 1) % self._maxsize
            self._index_ready_for_read[index_read].wait()
            self._index_ready_for_read[index_read].clear()

        nested_array = tree.map_structure(
            lambda x: np.copy(x[index_read]),
            self._nested_shared_numpy_array,
        )
        self._index_ready_for_write[index_read].set()
        return nested_array

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        del state["_nested_shared_numpy_array"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._nested_shared_numpy_array = self._get_nested_shared_numpy_array()

    def _get_nested_shared_numpy_array(self) -> NestedNumpyArray:
        return tree.map_structure(
            lambda array, shm: np.ndarray(
                array.shape, dtype=array.dtype, buffer=shm.buf
            ),
            self._original_nested_array,
            self._nested_shared_memory_space,
        )
