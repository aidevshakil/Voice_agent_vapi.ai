"""Core: configuration, logging, errors, dependency container.

Intentionally free of re-exports — ``Container`` pulls in the whole RAG stack, so
importing it from the package root would make ``app.core`` a heavy, cycle-prone
import. Import from the submodules directly (``from app.core.container import
Container``).
"""
