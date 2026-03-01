"""
SLiCEs: Structured Linear CDE recurrences for PyTorch.

This package provides efficient structured linear controlled differential equation
(SLiCE) recurrences for sequence modelling.

Main components:
    - SLiCE: Core structured recurrence
    - SLiCELayer: Default residual SLiCE layer
    - StackedSLiCE: Stacked model with embedding and output projection
"""

from slices.slices import SLiCE, SLiCELayer, StackedSLiCE

__version__ = "0.2.0"
__all__ = ["SLiCE", "SLiCELayer", "StackedSLiCE"]
