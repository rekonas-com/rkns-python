from pathlib import Path


def detect_format(path: Path) -> str | None:
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
    if path.suffix.lower() != ".edf":
        return None

    # Read EDF version from ASCII Character at the beginning of file
    # see https://www.edfplus.info/specs/edf.html
    with path.open("rb") as file:
        first_byte = file.read(8)

    # Decode the bytes as an ASCII string
    edf_version = first_byte.decode("ascii").strip()

    fileformat = {"0": "EDF", "1": "EDF_PLUS"}

    return fileformat.get(edf_version, None)
