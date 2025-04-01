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
        """Apply transformation to requested elements.
        Just forward the slicing logic to source object.
        """
        source_sliced = cast(np.ndarray, self._source.__getitem__(idx))
        return self._transform(source_sliced, idx)

    def _transform(
        self, chunk: np.ndarray, idx: Union[int, slice, Tuple, np.ndarray]
    ) -> np.ndarray:
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
        self,
        source: zarr.Array,
        pmin: np.ndarray,
        pmax: np.ndarray,
        dmin: np.ndarray,
        dmax: np.ndarray,
    ):
        super().__init__(source)
        self._m = (pmax - pmin) / (dmax - dmin)
        self._bias = (pmax / self._m) - dmax

        if len(self._m.shape) != 2 or self._m.shape[1] != source.shape[1]:
            raise ValueError(
                f"Shape of scale factor {self._m.shape} does not match source shape {source.shape}."
            )

        if len(self._bias.shape) != 2 or self._bias.shape[1] != source.shape[1]:
            raise ValueError(
                f"Shape of scale factor {self._m.shape} does not match source shape {source.shape}."
            )

    def slice_transform_param(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slice the parameter self._m and self._bias, which are of shape [1, n_channels],
        as they are intended to be broadcastable to the data.
        This function should ignore row indexing, and apply channel indexing.

        Args:
            idx (_type_): index that is applied to the data, can be any slicing operation passed to __getitem__

        Returns:
            Tuple[np.ndarray, np.ndarray]: Sliced self._m, self._bias
        """
        if isinstance(idx, int) or isinstance(idx, slice):
            # Single integer or single slices, indexes only row, return all params
            return self._m, self._bias
        elif isinstance(idx, tuple):
            # Handle tuples
            if len(idx) > 2:
                raise IndexError("Too many indices for 2D array")
            elif len(idx) == 1:
                return self._m, self._bias
            # len(idx) == 2
            row, col = idx  # type: ignore
            if row is ...:
                row = slice(None)
            if col is ...:
                col = slice(None)

            if isinstance(row, int) and isinstance(col, int):
                # Single element. data will be a scalar, so _m and _bias should be as well.
                return self._m[0, col], self._bias[0, col]
            elif isinstance(row, int):
                # Single row, column slice
                return self._m[0, col], self._bias[0, col]
            elif isinstance(col, int):
                # Single column, row slice
                return self._m[:, col], self._bias[:, col]
            else:
                # Both row and column slices
                return self._m[:, col], self._bias[:, col]
        else:
            raise TypeError("Invalid index type")

    @staticmethod
    def is_full_slice(s: Union[slice, int, None]) -> bool:
        if isinstance(s, slice):
            return s.start is None and s.stop is None and s.step is None
        return False

    def _transform(
        self, chunk: np.ndarray, idx: Union[int, slice, Tuple, np.ndarray]
    ) -> np.ndarray:
        """Apply transformation to requested elements.
        Handles slicing of _b when the second dimension is sliced (including integer indices).
        """
        # Extract the second dimension's slice info

        _m, _bias = self.slice_transform_param(idx)
        return _m * (chunk + _bias)
