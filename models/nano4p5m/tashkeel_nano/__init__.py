"""tashkeel_nano — 4.5M-param Arabic diacritization model (HF: ahmadsy/tashkeel-nano-4p5m)."""
from .inference import TashkeelNano, strip_marks

__all__ = ["TashkeelNano", "strip_marks"]
__version__ = "1.0.0"
