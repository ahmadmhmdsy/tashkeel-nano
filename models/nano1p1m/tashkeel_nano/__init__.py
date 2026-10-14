"""tashkeel_nano - 1.1M-param Arabic diacritization model (HF: ahmadsy/tashkeel-nano-1p1m)."""
from .inference import TashkeelNano, strip_marks

__all__ = ["TashkeelNano", "strip_marks"]
__version__ = "1.0.0"
