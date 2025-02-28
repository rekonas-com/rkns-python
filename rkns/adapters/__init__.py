# adapter/__init__.py
from ..file_formats import FileFormat
from .registry import AdapterRegistry

# Register adapters by module path (lazy loading)
AdapterRegistry.register(FileFormat.EDF, "rkns.adapter.edf_adapter.RKNSEdfAdapter")
