import typing
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

X = TypeVar("X", bound=gym.spaces.Space)


class FunctionApprox(ABC, Generic[X]):
    """Abstract class for function approximation."""

    _param: NDArray[Any]
    domain: X

    @abstractmethod
    def value(self, x: Any) -> np.float64:
        """Evaluate the value of the function at x."""
        assert self.domain.contains(x), f"{x!r} ({type(x)}) invalid"
        return 0.0

    @abstractmethod
    def derivative(self, x: Any) -> NDArray[Any]:
        """Evaluate the value of the derivative at x."""
        assert self.domain.contains(x), f"{x!r} ({type(x)}) invalid"
        return np.zeros(self._param.shape)

    @property
    def param(self) -> NDArray[Any]:
        """Return the value of the protected attribute _param."""
        return self._param

    @param.setter
    def param(self, new_param: NDArray[Any]) -> None:
        """Update the value of the protected attribute _param."""
        assert new_param.shape == self._param.shape
        self._param = new_param

    def save(self, path: str) -> None:
        """Save the approximation function."""
        np.savetxt(fname=path, X=self._param)


class Tabular(FunctionApprox):
    """Abstract class for function approximation."""

    def __init__(
        self,
        d: typing.Tuple[gym.spaces.Discrete, ...] | gym.spaces.Discrete,
        param: NDArray[Any] | None = None,
    ) -> None:
        self.domain = d

        if isinstance(d, gym.spaces.Discrete):
            self._dim = (d.n,)
            self._start = self.domain.start
        else:
            self._dim = tuple(sp.n for sp in d)
            self._start = tuple([sp.start for sp in d])

        if param is None:
            self._param = np.zeros(self._dim)
        else:
            self._param = param

    def value(self, x: np.int64 | typing.Tuple[np.int64, ...]) -> np.float64:
        """The value of the function."""
        super().value(x)
        pos = np.empty(len(self._dim), dtype=np.int64)
        np.subtract(x, self._start, out=pos)
        return self._param[tuple(pos)]

    def derivative(self, x: np.int64) -> NDArray[Any]:
        """Differentiate of the function at x."""
        super().derivative(x)
        res = np.zeros(self._dim)
        pos = np.empty(len(self._dim), dtype=np.int64)
        np.subtract(x, self._start, out=pos)
        res[tuple(pos)] = 1.0
        return res


class UnBoundedBox(Exception):
    """Raised when the Box is unbounded."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

    pass


class TileCoding:
    """Construct tilings from a bounded box."""

    def __init__(
        self,
        box: gym.spaces.Box,
        n_tilings: int = 8,
        grid_size: tuple[int, ...] | None = None,
    ) -> None:
        """Create tilings."""
        if not box.is_bounded():
            raise UnBoundedBox
        self.box = box
        dim = np.prod(box.shape)

        if isinstance(grid_size, tuple):
            assert len(grid_size) == dim
        else:
            grid_size = dim * (n_tilings,)

        self.grid_size = grid_size

        self.tilings = list()
        for i in range(n_tilings):
            grid = np.array(
                [
                    np.linspace(low, high, n + 1)[1:-1]
                    + i * (high - low) / n / n_tilings
                    for low, high, n in zip(
                        self.box.low.flatten(), self.box.high.flatten(), grid_size
                    )
                ]
            )
            self.tilings.append(grid)

    def tile_encode(self, sample: NDArray) -> NDArray:
        """Encode given sample using tile-coding.

        Parameters
        ----------
        sample : array_like
            A single sample from the (original) continuous space.
        flatten : bool
            If true, flatten the resulting binary arrays into a single long vector.

        Returns
        -------
        encoded_sample : list or array_like
            A list of binary vectors, one for each tiling, or flattened into one.
        """
        assert self.box.contains(sample)
        encoded_sample = np.empty(
            (len(self.tilings), len(self.grid_size) + 1), dtype=int
        )
        for i, grid in enumerate(self.tilings):
            encoded_sample[i, 0] = i
            encoded_sample[i, 1:] = np.array(
                [np.digitize(s, g) for s, g in zip(sample.flatten(), grid)]
            )
        return encoded_sample.T


class LinearTiling(FunctionApprox[gym.spaces.Box]):
    """Approximate function using Tile coding and linear function."""

    def __init__(self, t: TileCoding, param: NDArray[Any] | None = None) -> None:
        self.domain = t.box
        self.tilecode = t

        if param is None:
            self._param = np.zeros((len(t.tilings),) + t.grid_size)
        else:
            self._param = param

    def value(self, x: Any) -> np.float64:
        """The value of the function."""
        super().value(x)
        return self._param[tuple(self.tilecode.tile_encode(x))].sum()

    def derivative(self, x: Any) -> NDArray[Any]:
        """Differentiate of the function at x."""
        super().derivative(x)
        res = np.zeros((len(self.tilecode.tilings),) + self.tilecode.grid_size)
        res[tuple(self.tilecode.tile_encode(x))] = 1.0
        return res


if __name__ == "__main__":
    t = TileCoding(gym.spaces.Box(-1.0, 1.0, (2,)), n_tilings=8)
    f = LinearTiling(t)
    print(t.tile_encode(np.zeros((2,), dtype=np.float32)))
    print(f.param.shape)
    y = f.value(np.float32(np.random.random((2,))))
    print(y)
    f.param += 1.0
    print(f.param)
    env = gym.make("Blackjack-v1")
    tab = Tabular(env.observation_space)
