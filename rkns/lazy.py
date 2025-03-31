from typing import Any, Generic, Tuple, TypeVar, Union, cast

import numpy as np
import zarr

T = TypeVar("T", bound=Union[zarr.Array, np.ndarray])


class LazyIndexer(Generic[T]):
    """
    Generic lazy-evaluated container that applies transformations on-demand.
    Supports all NumPy indexing patterns while avoiding premature materialization.

    Usage:
    - Inherit and implement `_transform` for specific conversions
    - Wrap any array-like object (zarr, numpy, etc.)
    """

    def __init__(self, source: T):
        self._source = source

    def __getitem__(self, idx: Union[int, slice, Tuple, np.ndarray]) -> np.ndarray:
        """Apply transformation to requested elements"""
        source_sliced = cast(np.ndarray, self._source[idx])
        return self._transform(source_sliced)

    def _transform(self, chunk: np.ndarray) -> np.ndarray:
        """Override this with domain-specific logic"""
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._source.shape

    @property
    def dtype(self) -> np.dtype:
        return np.result_type(self._source.dtype, np.float32)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        arr = self[:]  # Materialize full array
        return arr.astype(dtype) if dtype else arr

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape}, dtype={self.dtype})"


class LazySignal(LazyIndexer[zarr.Array]):
    """EDF-format signal specific implementation"""

    def __init__(
        self, source: zarr.Array, pmin: float, pmax: float, dmin: float, dmax: float
    ):
        super().__init__(source)
        self._m = (pmax - pmin) / (dmax - dmin)
        self._bias = (pmax / self._m) - dmax

    def _transform(self, chunk: np.ndarray) -> np.ndarray:
        return self._m * (chunk + self._bias)
