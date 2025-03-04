from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..file_formats import FileFormat

if TYPE_CHECKING:
    from zarr.storage import StoreLike


def detect_format(path: StoreLike) -> FileFormat:
    """
    If the suffix in lowercase is ".edf", this function detects edf and edf+
    based on the first byte of the file.

    Returns None if type was not successfully detected.

    Parameters
    ----------
    path
        File of interest.

    Returns
    -------
        "EDF", "EDF_PLUS", or None if not detected.
    """

    # Stores are not valid for edfs..
    if not isinstance(path, Path) and not isinstance(path, str):
        return FileFormat.UNKNOWN

    path = Path(path)
    if path.suffix.lower() != ".edf":
        return FileFormat.UNKNOWN

    # Read EDF version from ASCII Character at the beginning of file
    # see https://www.edfplus.info/specs/edf.html
    with path.open("rb") as file:
        first_byte = file.read(8)

    # Decode the bytes as an ASCII string
    edf_version = first_byte.decode("ascii").strip()
    fileformat = {"0": FileFormat.EDF, "1": FileFormat.EDF_PLUS}.get(
        edf_version, FileFormat.UNKNOWN
    )
    return fileformat
