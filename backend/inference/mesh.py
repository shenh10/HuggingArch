"""Compatibility shim — the mesh is `arch.layout.mesh`.

Layout already consumes the rank grid; keeping a second copy under
`inference` was the upward import the walk could not justify.
"""
from backend.arch.layout.mesh import *  # noqa: F403
from backend.arch.layout.mesh import (  # noqa: F401 — explicit for re-importers
    GLOBAL_OUTER_AXES,
    REGION_AXES,
    SUBGROUP_AXES,
    axis_degrees,
    base_stride,
    group_size,
    region_of,
    spans,
    subgroup_degrees,
    subgroup_stride,
)
