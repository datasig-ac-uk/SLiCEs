"""
SLiCEs: Structured Linear CDE layers for PyTorch.

This package provides efficient structured linear controlled differential equation
(SLiCE) layers for sequence modelling.

Main components:
    - SLiCE: Core structured linear recurrence layer
    - SLiCEBlock: Residual block wrapping SLiCE with post-activation
    - StackedSLiCE: Stacked model with embedding and output projection
"""

from slices.slices import SLiCE, SLiCEBlock, StackedSLiCE

__version__ = "0.1.0"
__all__ = ["SLiCE", "SLiCEBlock", "StackedSLiCE"]
