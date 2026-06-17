"""
Vietnamese Automatic License Plate Recognition (ALPR) System.

A two-stage detection pipeline using YOLOv8 for Vietnamese license plates:
  Stage 1: Plate detection (YOLOv8n)
  Stage 2: Character recognition (YOLOv8s, 35 classes)
  Post-processing: Row grouping + Vietnamese format correction

Author: izanw
"""

__version__ = "1.0.0"


def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies at package import time."""
    if name == "ALPRPipeline":
        from src.pipeline import ALPRPipeline
        return ALPRPipeline
    if name == "VietnamesePlatePostprocessor":
        from src.postprocessor import VietnamesePlatePostprocessor
        return VietnamesePlatePostprocessor
    raise AttributeError(f"module 'src' has no attribute {name!r}")


__all__ = ["ALPRPipeline", "VietnamesePlatePostprocessor"]
