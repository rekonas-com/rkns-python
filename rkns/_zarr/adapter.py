"""
This file dispatches the zarr
"""

import zarr

from .utils_interface import ZarrUtils

is_zarr_v2 = zarr.__version__.startswith("2.")


def get_utils_implementation() -> type[ZarrUtils]:
    """Get the correct Zarr implementation based on version."""
    if is_zarr_v2:
        from .utils_zarr_v2 import _ZarrV2Utils

        return _ZarrV2Utils
    else:
        from .utils_zarr_v3 import _ZarrV3Utils

        return _ZarrV3Utils


# def get_storehandler_implementation() -> None:
#     """Get the correct Zarr implementation based on version."""
if is_zarr_v2:
    from .storehandler_zarr_v2 import StoreHandlerZarrV2 as StoreHandler
    # return StoreHandlerZarrV2
else:
    from .storehandler_zarr_v3 import StoreHandlerZarrV3 as StoreHandler


# Note: While this is technically a class that is returned,
# we still use lower-case to indicate that it is to be used as a module.
zarr_utils = get_utils_implementation()
# storehandler on the other hand is to be used as a class.
# StoreHandler = get_storehandler_implementation()

__all__ = ["zarr_utils", "StoreHandler", "is_zarr_v2"]
