from ..file_formats import FileFormat
from .registry import FileFormatRegistry

# Register adapters by module path (lazy loading)
FileFormatRegistry.register_detector(
    FileFormat.EDF, "rkns.detectors.edf_detector.detect_format"
)
