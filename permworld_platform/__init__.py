"""PermWorld data platform: exact versioned task interfaces and generators."""

from .objects import canonical_orbit_key, transform, validate_permutation
from .tasks import label

__version__ = "0.1.0"
__all__ = ["label", "validate_permutation", "transform", "canonical_orbit_key"]
