import logging
import math
import multiprocessing
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import numpy as np
import tree

from fastpbrl.types import Transition


@dataclass
class ReplayBufferConfig:
    min_size_to_sample: int = 25_000  # Minimum number of transitions in the replay
    # buffer to allow sampling from it.
    max_replay_buffer_size: int = 1_000_000  # Maximum number of transitions stored in
    # the replay buffer, transitions are deleted in a FIFO fashion afterwards
    samples_per_insert: Optional[float] = None  # Target ratio for number of transitions
    # sampled per transition inserted in the replay buffer. If None, no constraint is
    # imposed on this ratio.
    insert_error_buffer: Optional[float] = None  # in number of transitions.
    # This parameter allows to deviate from the targeted sample_per_insert ratio by some
    # buffer.
    # Specifically |(nb_inserts - min_size_to_sample) - nb_samples / samples_per_insert|
    # is allowed to be as big as insert_error_buffer.
    # Increasing insert_error_buffer de-synchronizes more the actors and the learner
    # but it is more efficient as the actors and learners do not have to wait for each
    # other after every insert/sample.
    # If insert_error_buffer is None, no constraint is imposed on the ratio.

    def __post__init__(self):
        assert self.min_size_to_sample > 0
        assert self.max_replay_buffer_size >= self.min_size_to_sample
        if self.samples_per_insert is not None:
            assert self.samples_per_insert > 0.0
            assert self.insert_error_buffer is not None


class ReplayBuffer:
    """Basic implementation of a replay buffer stored in memory.

    Transition data is stored in numpy arrays (one numpy array for each member variable
    of the Transition type) which are pre-allocated to the maximum possible size of the
    replay buffer the first time data is added.

    Samples are discarded once the replay buffer has reached its maximum size in a FIFO
    fashion.
    """

    def __init__(
        self,
        max_size: int,
    ):
        """
        Parameters:
            max_size (int): maximum number of Transitions stored in the replay buffer.
        """

        self._max_size = max_size
        self._random_generator = np.random.default_rng()
        self._row_index = 0
        self._are_all_entries_populated = False
        self._dataset = None

    def add(self, data_batch: Transition):
        """
        Add a batch of Transitions to the replay buffer

        Parameters:
            data_batch (Transition): A batch of transition data. The first dimension of
                each member variable of the data_batch should be the batch dimension.
        """

        if self._dataset is None:
            # Allocate the memory for the full dataset the first time we get a data
            # batch
            self._dataset = tree.map_structure(
                lambda x: np.zeros(
                    shape=tuple([self._max_size] + list(x.shape[1:])),
                    dtype=x.dtype,
                ),
                data_batch,
            )

        num_new_samples = data_batch.reward.shape[0]

        if num_new_samples > self._max_size:
            # There is no point in adding all the new samples
            # as some of them will be immediately deleted so keep
            # only the ones that will persist.
            self._are_all_entries_populated = True
            num_new_samples = self._max_size
            data_batch = tree.map_structure(lambda x: x[-num_new_samples:], data_batch)

        # Identify the continuous slices of batches to copy over to the dataset.
        # There will be two if we add more samples than we can fit starting at the
        # current row index, otherwise they will just be one.
        slices_to_copy = []
        start_index_data_batch = 0
        if self._row_index + num_new_samples >= self._max_size:
            self._are_all_entries_populated = True
            slices_to_copy.append((self._row_index, self._max_size, 0))
            num_new_samples -= self._max_size - self._row_index
            start_index_data_batch = self._max_size - self._row_index
            self._row_index = 0

        slices_to_copy.append(
            (
                self._row_index,
                self._row_index + num_new_samples,
                start_index_data_batch,
            )
        )

        self._row_index = (self._row_index + num_new_samples) % self._max_size

        flattened_data_batch = dict(tree.flatten_with_path(data_batch))
        flattened_dataset = dict(tree.flatten_with_path(self._dataset))

        # Now copy over the data
        for (
            start_index_replay_buffer,
            end_index_replay_buffer,
            start_index_data_batch,
        ) in slices_to_copy:
            for path, tensor in flattened_dataset.items():
                end_index_data_batch = (
                    start_index_data_batch
                    + end_index_replay_buffer
                    - start_index_replay_buffer
                )
                tensor[
                    start_index_replay_buffer:end_index_replay_buffer
                ] = flattened_data_batch[path][
                    start_index_data_batch:end_index_data_batch
                ]

    def sample(self, batch_size: int) -> Transition:
        """
        Sample a batch of transitions uniformly at random in the replay buffer.

        Parameters:
            batch_size (int): number of Transitions to return.

        Return:
            Sampled transitions are returned in the form of a single Transition
                object with its member variable stacked along the batch dimension
                (i.e. the first index of the shape of all member variables is
                 batch_size)
        """

        assert self._row_index >= 1 or self._are_all_entries_populated
        indices = self._random_generator.integers(
            low=0,
            high=(
                self._max_size if self._are_all_entries_populated else self._row_index
            ),
            size=batch_size,
        )
        return tree.map_structure(lambda x: x[indices], self._dataset)

    def size(self) -> int:
        """
        Get the number of transitions currently stored in the replay buffer.

        Return:
            Number of transitions currently stored in the replay buffer.
        """

        if self._are_all_entries_populated:
            return self._max_size
        return self._row_index

    def flush(self):
        """
        Discard all the transitions currently stored in the replay buffer.
        """
        self._row_index = 0
        self._are_all_entries_populated = False


class ReplayBufferWithSampleRatio(Iterable[Transition]):
    """Wrapper around one/multiple independent ReplayBuffers to control
    sampling / insertion flow in the same spirit as done in the reverb library
    https://github.com/deepmind/reverb but with the replay buffered stored in-memory.

    Controling the sampling / insertion flow is useful in cases when interactions
    with the environments happen in multiple processes in parallel for performance
    purposes, in which case transition data points are added to queues but we still
    need to control precisely the ratio of update steps per environment steps.

    For each queue of Transition objects provided at instantiation, an independent
    replay buffer is created and continuously populated with transition batches
    retrieved from that queue in the background.

    Batches of transitions obtained from the replay buffer can be iterated on by
    constructing an iterator of this object:
    ```
    for data_batch in replay_buffer_with_sample_ratio:
        ...
    ```
    """

    def __init__(
        self,
        list_transition_queue: List[multiprocessing.Queue],
        batch_size: int,
        config: ReplayBufferConfig,
    ):
        """
        Parameters:
            list_transition_queue (List[multiprocessing.Queue]): list of queues that are
                continuously populated with Transition objects
            batch_size (int): Transition objects returned when iterating over this
                object will correspond to batches of size batch_size (one for each
                queue provided in list_transition_queue).
            config (ReplayBufferConfig): configuration that specifies the sampling /
                insertion control flow.
        """
        assert batch_size > 0

        self._list_transition_queue = list_transition_queue
        self._batch_size = batch_size
        self._min_size_to_sample = config.min_size_to_sample
        self._samples_per_insert = config.samples_per_insert
        self._sample_error_buffer = (
            config.insert_error_buffer * config.samples_per_insert
            if config.samples_per_insert is not None
            else None
        )

        if self._samples_per_insert is not None and self._sample_error_buffer < max(
            1.0, self._samples_per_insert
        ):
            raise ValueError(
                "The spread allowed by insert_error_buffer is too small "
                "and could lead to completely block sampling and/or inserting "
                "calls."
            )

        self._num_inserts = 0
        self._num_samples = 0
        self._list_replay_buffer = [
            ReplayBuffer(
                max_size=config.max_replay_buffer_size,
            )
            for _ in list_transition_queue
        ]
        self._logger = logging.getLogger(f"{__name__}")
        self._just_sampled = threading.Event()
        self._just_inserted = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._background_fetch_data, args=(), daemon=True
        )
        self._thread.start()

    def __iter__(self) -> Iterator[List[Transition]]:
        """
        Generates an iterator that returns batches of transition data sampled
        independently from the replay buffer corresponding to each queue provided
        at instantiation.

        Return:
            A List of length len(list_transition_queue) of Transition objects
                with their member variables with shapes (batch_size, ...)
                (i.e. a batch of transition data for each replay buffer).
        """

        time_start_get_batch = None
        while True:
            # Wait until the rate limiter allows us to sample
            # a new batch.
            list_data_batch: Optional[List[List[Transition]]] = None
            nb_samples_in_batch = 0
            while True:
                with self._lock:
                    max_nb_samples_allowed = self._max_nb_samples_allowed()
                    if max_nb_samples_allowed == 0:
                        pass
                    elif list_data_batch is None:
                        list_data_batch = [
                            [replay_buffer.sample(max_nb_samples_allowed)]
                            for replay_buffer in self._list_replay_buffer
                        ]
                        nb_samples_in_batch += max_nb_samples_allowed
                        self._num_samples += max_nb_samples_allowed
                        self._just_sampled.set()
                    else:
                        nb_samples_to_add = min(
                            self._batch_size - nb_samples_in_batch,
                            max_nb_samples_allowed,
                        )

                        for i, replay_buffer in enumerate(self._list_replay_buffer):
                            list_data_batch[i].append(
                                replay_buffer.sample(nb_samples_to_add)
                            )

                        nb_samples_in_batch += nb_samples_to_add
                        self._num_samples += nb_samples_to_add
                        self._just_sampled.set()

                    if nb_samples_in_batch == self._batch_size:
                        break

                    self._just_inserted.clear()
                # Nothing will change until the background thread adds
                # data to the replay buffer so we can wait here until we get
                # the signal that some data points have been added.
                self._just_inserted.wait()

            if time_start_get_batch is not None:
                self._logger.debug(
                    f"Time to get batch: {time.time() - time_start_get_batch}"
                )
            list_data_batch_concatenated = [
                tree.map_structure(
                    lambda *x: np.concatenate(x, axis=0),
                    *data_batch,
                )
                for data_batch in list_data_batch
            ]

            yield list_data_batch_concatenated
            time_start_get_batch = time.time()

    def flush(self):
        """
        Discard all the transitions currently stored in the replay buffers.
        """
        with self._lock:
            for replay_buffer in self._list_replay_buffer:
                replay_buffer.flush()
            self._just_sampled.set()
            self._just_inserted.set()

    def _max_nb_inserts_allowed(self) -> int:
        """
        Return:
            Maximum number of transitions that can be added to each replay buffer
            given the constraints specified in the replay buffer config and
            the total the number of transitions sampled so far from __iter__.
        """
        replay_buffer_size = self._list_replay_buffer[0].size()
        if self._samples_per_insert is None:
            return np.inf

        nb_inserts = self._num_inserts
        if replay_buffer_size < self._min_size_to_sample:
            # We will add at least self._min_size_to_sample - replay_buffer_size
            # data points but let's check how many more we can add without
            # violating the sample_to_insert ratio too much.
            nb_inserts += self._min_size_to_sample - replay_buffer_size

        nb_inserts_dev = nb_inserts - self._num_samples / self._samples_per_insert
        nb_inserts_allowed = max(
            0,
            math.ceil(
                self._min_size_to_sample
                + self._sample_error_buffer / self._samples_per_insert
                - nb_inserts_dev
            ),
        )

        if replay_buffer_size < self._min_size_to_sample:
            return (self._min_size_to_sample - replay_buffer_size) + nb_inserts_allowed
        return nb_inserts_allowed

    def _max_nb_samples_allowed(self) -> int:
        """
        Return:
            Maximum number of transitions that can be sampled from each
            replay buffer given the constraints specified in the replay buffer
            config and the total the number of transitions retrieved from the queues
            so far.
        """

        if self._list_replay_buffer[0].size() < self._min_size_to_sample:
            return 0
        if self._samples_per_insert is None:
            return self._batch_size

        nb_samples_dev = (
            self._num_inserts * self._samples_per_insert - self._num_samples
        )
        nb_samples_allowed = max(
            0,
            math.floor(
                nb_samples_dev
                - self._min_size_to_sample * self._samples_per_insert
                + self._sample_error_buffer
            ),
        )
        return min(nb_samples_allowed, self._batch_size)

    def _background_fetch_data(self):
        """
        Continuously retrieve transition data from the queues provided at
        instantiation and add them to the corresponding replay buffers.
        """
        list_data_batch: List[Transition] = None

        while True:
            if list_data_batch is None:
                list_data_batch = [
                    transition_queue.get()
                    for transition_queue in self._list_transition_queue
                ]

            while True:
                # Wait until the rate limiter allows us to insert
                # new data points into the local replay buffer.
                with self._lock:
                    nb_inserts_allowed = self._max_nb_inserts_allowed()
                    if nb_inserts_allowed > 0:
                        break
                    self._just_sampled.clear()
                # Nothing will change until somebody samples from
                # the replay buffer so we can wait here until we get
                # the signal that somebody sampled some data.
                self._just_sampled.wait()

            nb_data_points_in_batch = list_data_batch[0].reward.shape[0]
            if nb_inserts_allowed < nb_data_points_in_batch:
                list_data_batch_to_add = [
                    tree.map_structure(lambda x: x[:nb_inserts_allowed], data_batch)
                    for data_batch in list_data_batch
                ]
                list_data_batch = [
                    tree.map_structure(lambda x: x[nb_inserts_allowed:], data_batch)
                    for data_batch in list_data_batch
                ]
            else:
                list_data_batch_to_add = list_data_batch
                list_data_batch = None

            with self._lock:
                for replay_buffer, data_batch_to_add in zip(
                    self._list_replay_buffer, list_data_batch_to_add
                ):
                    replay_buffer.add(data_batch_to_add)
                self._num_inserts += min(nb_data_points_in_batch, nb_inserts_allowed)
                self._just_inserted.set()
