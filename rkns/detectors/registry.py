from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from rkns.util import import_from_string

from ..file_formats import FileFormat

if TYPE_CHECKING:
    from typing import Callable

    from zarr.storage import StoreLike

    FileFormatDetector = Callable[[StoreLike], FileFormat]


__all__ = ["FileFormatRegistry"]


class FileFormatRegistry:
    """
    Registry for inferring file formats based on file content or extensions.

    Note:
    -   The detectors DO NOT have to match 1-to-1 to the adapters.
        E.g. the EDF detector might be capable of detecting EDF, EDF+, BDF, ...,
        avoiding redundant operations.
    -   The registered functions are always in the same order as the registration.
    -   The registered functions are lazily loaded in order of the registration.
    """

    _detector_fn_path: OrderedDict[FileFormat, str] = OrderedDict()

    @classmethod
    def register_detector(cls, format_name: FileFormat, detector_fn_path: str) -> None:
        """
        Register a function via its dotted path.

        Parameters
        ----------
        format_name
            Format name, just for future reference to identify the function.
        detector_fn_path
            Absolute path to the function.
        """
        cls._detector_fn_path[format_name] = detector_fn_path

    @classmethod
    def get_detector(cls, format_name: FileFormat) -> FileFormatDetector:
        """
        Dynamically load the detector function based on the format name.

        Note: This function might not be needed..

        Parameters
        ----------
        format_name
            Identifyer of the registered functions.

        Returns
        -------
            The callable for detecting the file format.

        Raises
        ------
        ValueError
            _description_
        """
        detector_path = cls._detector_fn_path.get(format_name)
        if not detector_path:
            raise ValueError(f"No detector found for {format_name=}")

        return import_from_string(detector_path)

    @classmethod
    def detect_fileformat(cls, file_path: StoreLike) -> FileFormat:
        """
        Detect the file format by iterating through all registered
        detector function until a match is found.

        Parameters
        ----------
        path
            Path to the file of interest.

        Returns
        -------
            An Enum identifying the detected format, e.g. "EDF".
            Returns FileFormat.UNKNOWN if the format could not be detected.
        """
        detected_format = FileFormat.UNKNOWN
        for format_name, detector_fn_dotted_path in cls._detector_fn_path.items():
            detector_fn: FileFormatDetector = import_from_string(
                detector_fn_dotted_path
            )
            detected_format = detector_fn(file_path)
            if detected_format != FileFormat.UNKNOWN:
                return detected_format
        return detected_format
