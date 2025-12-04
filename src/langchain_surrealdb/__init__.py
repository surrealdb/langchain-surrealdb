from importlib import metadata

from langchain_surrealdb.vectorstores import SurrealDBVectorStore

try:
    __version__ = metadata.version(__package__) if __package__ else ""
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "SurrealDBVectorStore",
    "__version__",
]
