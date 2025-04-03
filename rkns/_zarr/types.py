from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Iterable, Literal

############################################################
# Types taken from zarr v3
# https://github.com/zarr-developers/zarr-python/blob/main/src/zarr/core/common.py#L28
# The MIT License (MIT)
# Copyright (c) 2015-2024 Zarr Developers <https://github.com/zarr-developers>
BytesLike = bytes | bytearray | memoryview
ShapeLike = tuple[int, ...] | int
ChunkCoords = tuple[int, ...]
ChunkCoordsLike = Iterable[int]
ZarrFormat = Literal[2, 3]
NodeType = Literal["array", "group"]
JSON = str | int | float | Mapping[str, "JSON"] | Sequence["JSON"] | None
MemoryOrder = Literal["C", "F"]
AccessModeLiteral = Literal["r", "r+", "a", "w", "w-"]

if TYPE_CHECKING:
    try:
        # v3
        from zarr.abc.codec import BaseCodec as CodecType  # type: ignore # noqa: F401
    except ImportError:
        # v2
        from numcodecs.abc import Codec as CodecType  # type: ignore  # noqa: F401
