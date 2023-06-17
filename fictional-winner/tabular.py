import typing

import gymnasium as gym
import numpy as np
from core.fun import FunctionApprox
from numpy.typing import NDArray


class Tabular(FunctionApprox):
    """Abstract class for function approximation."""

    def __init__(
        self,
        d: typing.Tuple[gym.spaces.Discrete, ...] | gym.spaces.Discrete,
        param: NDArray[np.float64] | None = None,
    ) -> None:
        if isinstance(d, gym.spaces.Discrete):
            self._dim = (d.n,)
            self._start = d.start
        else:
            self._dim = tuple(sp.n for sp in d)
            self._start = tuple([sp.start for sp in d])

        if param is None:
            super().__init__(d, np.zeros(self._dim))
        else:
            assert self._dim == param.shape
            super().__init__(d, param)

    def value(self, x: np.int64 | typing.Tuple[np.int64, ...]) -> np.float64:
        """The value of the function."""
        super().value(x)
        pos = np.empty(len(self._dim), dtype=np.int64)
        np.subtract(x, self._start, out=pos)
        return self._param[tuple(pos)]

    def derivative(self, x: np.int64) -> NDArray[np.float64]:
        """Differentiate of the function at x."""
        super().derivative(x)
        res = np.zeros(self._dim)
        pos = np.empty(len(self._dim), dtype=np.int64)
        np.subtract(x, self._start, out=pos)
        res[tuple(pos)] = 1.0
        return res
