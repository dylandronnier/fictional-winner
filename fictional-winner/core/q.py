from __future__ import annotations

from typing import Any, Type

import gymnasium as gym
import numpy as np
from core.fun import *

X = TypeVar("X", bound=gym.spaces.Space)


class Q(FunctionApprox[gym.spaces.Tuple]):
    """A tuple (more precisely: the cartesian product) of :class:`FunctionApprox` instances.

    Elements of this space are tuples of elements of the constituent function
    approximations.

    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: X,
        hypothesis_set: Type[FunctionApprox[X]],
        param: NDArray[np.float64] | None = None,
    ):
        """Constructor of :class:`Tuple` function.

        Args:
        ----
            spaces (Iterable[FunctionApprox]): The spaces that are involved in the cartesian product.
        """
        self.domain = gym.spaces.Tuple((observation_space, action_space))
        shap = hypothesis_set(observation_space).param.shape
        self._fun = dict()
        if param is None:
            self._param = np.zeros((action_space.n,) + shap)
        else:
            self._param = param
        for i in range(action_space.start, action_space.n):
            self._fun[i] = hypothesis_set(
                observation_space, self._param[i - action_space.start]
            )

    def value(self, x: Any) -> np.float64:
        super().value(x)
        return self._fun[x[1]].value(x[0])

    def derivative(self, x: Any) -> NDArray[np.float64]:
        super().derivative(x)
        res = np.zeros(self._param.shape)
        res[x[1] - self.domain[1].start] = self._fun[x[1]].derivative(x[0])
        return res


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")
    f = R(env.action_space, env.observation_space, Tabular)
    print(f.param)
    f.param[2] += 3.0
    print(f.param)
    print(f._fun[2].param)
