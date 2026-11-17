"""Whether a walk binds runtime contracts declared by its components."""
from enum import Enum


class RuntimeProfile(str, Enum):
    ARCHITECTURE = 'architecture'
    DECLARED = 'declared'
