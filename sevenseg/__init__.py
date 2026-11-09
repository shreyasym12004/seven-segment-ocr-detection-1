"""Seven-segment digital meter OCR pipeline (classical CV, no model weights needed)."""
from .pipeline import read_image, read_meter

__all__ = ["read_meter", "read_image"]
