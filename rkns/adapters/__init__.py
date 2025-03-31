# adapter/__init__.py
from ..file_formats import FileFormat
from .registry import AdapterRegistry

# Register adapters by module path (lazy loading)
AdapterRegistry.register_adapter(
    FileFormat.EDF, "rkns.adapters.edf_adapter.RKNSEdfAdapter"
)
