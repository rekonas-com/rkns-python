try:
    # v3
    from zarr import Group as ZarrGroup  # type: ignore
except ImportError:
    # v2
    from zarr.hierarchy import Group as ZarrGroup  # type: ignore  # noqa: F401

from zarr import Array as ZarrArray  # type: ignore  # noqa: F401
