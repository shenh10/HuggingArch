"""Service HTTP routes registered against the shared FastAPI app.

Each submodule registers its endpoints via `@app.<verb>(...)` at import
time, mirroring the side-effect pattern used by `backend.api.routes`.
"""
from . import account    # noqa: F401
from . import admin      # noqa: F401
from . import analytics  # noqa: F401
from . import auth_session  # noqa: F401
from . import describe   # noqa: F401
from . import meta       # noqa: F401
from . import modules    # noqa: F401
from . import spec       # noqa: F401

__all__ = []
