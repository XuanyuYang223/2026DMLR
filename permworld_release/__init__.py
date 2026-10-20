"""Release-facing helpers that preserve the frozen platform implementation.

The installed release includes mathematical certificates, but not historical
research input pools. Data generated through this entry point must explicitly
declare that it is outside those experiments.
"""
from .data import generate_dataset, verify_dataset

__all__ = ["generate_dataset", "verify_dataset"]
