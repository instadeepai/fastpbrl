from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

import gym
import jax
import numpy as np
import tree
from jax import numpy as jnp
from tensorboardX import SummaryWriter

from fastpbrl.types import NestedArray, NestedJaxArray, NestedNumpyArray, Transition


def make_gym_env(env_name: str, rescale_action: bool = True) -> gym.Env:
    """Make a gym environment with a wrapper that transforms the observation in numpy
    array. Moreover, the action will be rescaled between -1 and 1 if the rescale_action
    boolean is set to True.


    Args:
        env_name (str): name of the environment.
        rescale_action (bool): boolean to enable the rescaling of the action between -1
            and 1.

    Returns:
        An environment wrapped with a transform observation wrapper. It may also be
            wrapped with a rescale action wrapper if the rescale_action is set to True.
    """
    environment = gym.wrappers.TransformObservation(
        env=gym.make(env_name),
        f=lambda observation: tree.map_structure(
            lambda x: x.astype(np.float32), observation
        ),
    )
    if rescale_action:
        return gym.wrappers.RescaleAction(env=environment, a=-1.0, b=1.0)
    else:
        return environment


def jax_tree_stack(tree_list: List[NestedJaxArray]) -> NestedJaxArray:
    """Transform a list of tree into a tree with the leaves stacked and
    cast into JAX arrays.

    Args:
        tree_list (NestedArray): list of nested array to process.

    Returns:
        A single nested jax array with the leaves stacked on the zeroth axis.
    """
    return jax.tree_util.tree_map(
        lambda *x: jnp.stack(x, axis=0),
        *tree_list,
    )


def numpy_tree_stack(tree_list: Sequence[NestedNumpyArray]) -> NestedNumpyArray:
    """Transform a list of tree into a tree with the leaves stacked and
    cast into numpy arrays.

    Args:
        tree_list (NestedArray): list of nested array to process.

    Returns:
        A single nested numpy array with the leaves stacked on the zeroth axis.
    """
    return jax.tree_util.tree_map(
        lambda *x: np.stack(x, axis=0),
        *tree_list,
    )


def write_dict(
    writer: SummaryWriter,
    tag_to_scalar_value: Dict[str, float],
    global_step: int,
    walltime: Optional[float] = None,
) -> None:
    """Loop through the tags and the scalar values to add them to the
    tensorboard summary writer.

    Args:
        writer (SummaryWriter):  tensorflow summary writer.
        tag_to_scalar_value (Dict[str, float]): mapping from tags to scalar values.
        global_step (int): step used for the abscissa of the tensorboard graph.
        walltime (Optional[float]): time used for the abscissa of the tensorboard graph.
    """
    for tag, scalar_value in tag_to_scalar_value.items():
        writer.add_scalar(
            tag=tag,
            scalar_value=scalar_value,
            global_step=global_step,
            walltime=walltime,
        )


def write_config(writer: SummaryWriter, config: Any) -> None:
    """Save the configuration in tensorboard as text.

    Args:
        writer (SummaryWriter):  tensorflow summary writer.
        config: A config object that possesses a `__dict__` attribute.
    """
    for k, v in config.__dict__.items():
        writer.add_text(tag=k, text_string=str(v))


def _index_select(
    source: np.ndarray,
    target: np.ndarray,
    source_index: int,
    target_index: int,
) -> np.ndarray:
    """Create a copy of a target array where the values at the target_index
    are replaced from the values of the source array from the source_index.

    Args:
        source (np.ndarray): source numpy array.
        target (np.ndarray): target numpy array.
        source_index (int):  source index.
        target_index (int): target index.

    Returns:
        `target` array with the values at `target_index` are replaced by the
        values of the `source` array at the index `source_index`
    """
    array = deepcopy(target)
    array[target_index] = source[source_index]
    return array


def tree_index_select(
    source: NestedArray, target: NestedArray, source_index: int, target_index: int
) -> NestedArray:
    """Create a copy of a target NestedArray where the values at the target_index
    are replaced from the values of the source NestedArray form the source_index.
    This function is useful to select the hyperparams and params during a PBT update.

    Args:
        source (NestedArray): tree source.
        target (NestedArray): tree target.
        source_index (int):  source index.
        target_index (int): target index.

    Returns:
        `target` tree with the leaves arrays possessing the values at `target_index`
        replaced by the values from the `source` tree leaves arrays at the index
        `source_index`
    """
    return jax.tree_map(
        lambda a, b: _index_select(a, b, source_index, target_index), source, target
    )


def _update(element: Any, target: np.ndarray, index: int) -> np.ndarray:
    """Update the target array using the index and the element.

    Args:
        element (Any): object to assign.
        target (np.ndarray): target array.
        index (int): index at which the element will be set.

    Returns:
        `target` array updated.
    """
    target[index] = element
    return target


def tree_update(
    source: NestedArray, target: NestedArray, target_index: int
) -> NestedArray:
    """Update the value at the index `index` of the target NestedArray using the source
    NestedArray index and the NestedArray source.

    The source should have a lower leaf dimension than the target.

    Args:
        source (NestedArray): tree source.
        target (NestedArray): tree target.
        target_index (int): target index.

    Returns:
        An updated version of target with the source tree set at the `target_index`
            index.
    """
    return jax.tree_map(lambda a, b: _update(a, b, target_index), source, target)


def unpack(stacked_tree: NestedJaxArray) -> List[NestedJaxArray]:
    """Unpack a NestedArray that has stacked leaves into a list of NestedArray.

    Args:
        stacked_tree (NestedArray): tree containing stacked leaves.

    Returns:
        List of trees.
    """
    size = jax.tree_leaves(stacked_tree)[0].shape[0]
    return [jax.tree_multimap(lambda x: x[i], stacked_tree) for i in range(size)]


def squeeze_last_axis(jax_array: jnp.ndarray) -> jnp.ndarray:
    return jnp.squeeze(jax_array, axis=-1) if jax_array.shape[-1] == 1 else jax_array


def fix_transition_shape(transition_batch: Transition) -> Transition:
    return transition_batch._replace(
        reward=squeeze_last_axis(transition_batch.reward),
        done=squeeze_last_axis(transition_batch.done),
    )


def polyak_averaging(x: jnp.ndarray, y: jnp.ndarray, tau: float) -> jnp.ndarray:
    return x * (1 - tau) + y * tau
