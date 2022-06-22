from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Union

import jax
import numpy as np
import tree
from jax import numpy as jnp

NestedJaxArray = tree.StructureKV[str, jnp.DeviceArray]
NestedNumpyArray = tree.StructureKV[str, np.ndarray]
NestedArray = Union[NestedJaxArray, NestedNumpyArray]

Action = jnp.DeviceArray
Observation = jnp.DeviceArray
Reward = jnp.DeviceArray
Done = jnp.DeviceArray


PRNGKey = jax.random.PRNGKey
HyperParams = NestedJaxArray
PolicyParams = NestedJaxArray
CriticParams = NestedJaxArray
EntropyParams = NestedJaxArray
SelectActionFunction = Callable[[PolicyParams, HyperParams, Observation, bool], Action]

NetworkOutput = NestedArray
QValues = jnp.ndarray
Logits = jnp.ndarray
Value = jnp.ndarray
SampleFn = Callable[[NetworkOutput, jax.random.PRNGKey], Action]
LogProbFn = Callable[[NetworkOutput, Action], Logits]


TrainingState = Any


class Transition(NamedTuple):
    observation: Observation
    action: Action
    reward: Reward
    done: Done
    next_observation: Observation


@dataclass
class Step:
    total_step_id: int
    episode_id: int
    step_id: int
    current_return: float
    is_episode_done: bool
    last_transition: Transition
