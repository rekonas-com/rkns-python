from typing import Any, Tuple, TypeVar, Union, cast

import numpy as np

from rkns.zarr.zarr_util import ZarrArray

T = TypeVar("T", bound=Union[ZarrArray, np.ndarray])


class LazySignal:
    """
    A lazy-evaluated signal container that applies transformations on-demand.

    This class is designed for signals that are stored similar to EDF (i.e. shifted and scaled) and supports NumPy-like indexing
    while avoiding premature materialization of the data. It applies scaling and bias
    transformations only when data is accessed.
    """

    def __init__(
        self, source: ZarrArray | np.ndarray, _m: np.ndarray, _bias: np.ndarray
    ):
        """
        Initialize the LazySignal with source data, scaling factors, and biases.

        Parameters
        ----------
        source : ZarrArray or np.ndarray
            The underlying data array to be transformed.
        _m : np.ndarray
            Scaling factors of shape (1, n_channels).
        _bias : np.ndarray
            Bias terms of shape (1, n_channels).
        """
        self._source = source
        self._m = _m
        self._bias = _bias

        if len(self._m.shape) != 2 or self._m.shape[1] != source.shape[1]:
            raise ValueError(
                f"Shape of scale factor {self._m.shape} does not match source shape {source.shape}."
            )

        if len(self._bias.shape) != 2 or self._bias.shape[1] != source.shape[1]:
            raise ValueError(
                f"Shape of scale factor {self._m.shape} does not match source shape {source.shape}."
            )

    def __getitem__(self, idx: Union[int, slice, Tuple, np.ndarray]) -> np.ndarray:
        """
        Retrieve transformed data for the given index or slice.

        Parameters
        ----------
        idx : int, slice, tuple, or np.ndarray
            The index or slice to apply to the data.

        Returns
        -------
        np.ndarray
            The transformed data corresponding to the requested slice.
        """
        source_sliced = cast(np.ndarray, self._source.__getitem__(idx))
        return self._transform(source_sliced, idx)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the underlying data.

        Returns
        -------
        tuple
            The shape of the source array.
        """
        return self._source.shape

    @property
    def dtype(self) -> np.dtype:
        """
        Get the dtype of the transformed data.

        Returns
        -------
        np.dtype
            The dtype after applying transformations.
        """
        return np.result_type(self._source.dtype, self._m.dtype)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """
        Convert the entire array to a NumPy array.

        Parameters
        ----------
        dtype : dtype, optional
            The desired dtype for the output array.

        Returns
        -------
        np.ndarray
            The fully materialized and transformed array.
        """
        arr = self[:]  # Materialize full array
        return arr.astype(dtype) if dtype else arr  # type: ignore

    def __repr__(self) -> str:
        """
        Return a string representation of the LazySignal.

        Returns
        -------
        str
            A string describing the shape and dtype of the signal.
        """
        return f"{type(self).__name__}(shape={self.shape}, dtype={self.dtype})"

    @classmethod
    def from_minmaxs(
        cls,
        source: ZarrArray | np.ndarray,
        pmin: np.ndarray,
        pmax: np.ndarray,
        dmin: np.ndarray,
        dmax: np.ndarray,
    ) -> "LazySignal":
        """
        Create a LazySignal from physical and digital min/max values.

        Parameters
        ----------
        source : ZarrArray or np.ndarray
            The source data array.
        pmin : np.ndarray
            Physical minimum values.
        pmax : np.ndarray
            Physical maximum values.
        dmin : np.ndarray
            Digital minimum values.
        dmax : np.ndarray
            Digital maximum values.

        Returns
        -------
        LazySignal
            A new LazySignal instance with computed scaling and bias.
        """
        _m = (pmax - pmin) / (dmax - dmin)
        _bias = (pmax / _m) - dmax
        return cls(source=source, _m=_m, _bias=_bias)

    def slice_columns_param(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slice the scaling and bias parameters based on indexing.

        Handles slicing of parameters to match the data indexing pattern.

        Parameters
        ----------
        idx : int, slice, or tuple
            The index or slice applied to the data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The sliced scaling (_m) and bias (_bias) parameters.
        """
        if isinstance(idx, int) or isinstance(idx, slice):
            return self._m, self._bias
        elif isinstance(idx, tuple):
            if len(idx) > 2:
                raise IndexError("Too many indices for 2D array")
            elif len(idx) == 1:
                return self._m, self._bias
            row, col = idx  # type: ignore
            if row is ...:
                row = slice(None)
            if col is ...:
                col = slice(None)

            if isinstance(row, int) and isinstance(col, int):
                return self._m[0, col], self._bias[0, col]
            elif isinstance(row, int):
                return self._m[0, col], self._bias[0, col]
            elif isinstance(col, int):
                return self._m[:, col], self._bias[:, col]
            else:
                return self._m[:, col], self._bias[:, col]
        else:
            raise TypeError("Invalid index type")

    @staticmethod
    def is_full_slice(s: Union[slice, int, None]) -> bool:
        """
        Check if a slice is a full slice ([:]).

        Parameters
        ----------
        s : slice, int, or None
            The slice to check.

        Returns
        -------
        bool
            True if the slice covers the entire range, False otherwise.
        """
        if isinstance(s, slice):
            return s.start is None and s.stop is None and s.step is None
        return False

    def _transform(
        self, data_slice: np.ndarray, idx: Union[int, slice, Tuple, np.ndarray]
    ) -> np.ndarray:
        """
        Apply scaling and bias transformation to the sliced data.

        Parameters
        ----------
        chunk : np.ndarray
            The data chunk to transform.
        idx : int, slice, tuple, or np.ndarray
            The index or slice used to retrieve the chunk.

        Returns
        -------
        np.ndarray
            The transformed data chunk.
        """
        _m, _bias = self.slice_columns_param(idx)
        return _m * (data_slice + _bias)
