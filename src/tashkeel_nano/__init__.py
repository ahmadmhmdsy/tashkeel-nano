"""tashkeel-nano - tiny Arabic diacritization (tashkeel) models.

    from tashkeel_nano import TashkeelNano
    model = TashkeelNano.from_pretrained("ahmadsy/tashkeel-nano-1p1m")
    print(model.diacritize("السلام عليكم"))

Import cost is deliberately split. The metric/label/codec layer is stdlib-only
and is imported eagerly; everything that needs torch is resolved lazily through
__getattr__, so scoring a prediction file on a CPU-only box never loads a
training stack.
"""
from __future__ import annotations

import importlib

# --- eager: stdlib only, no torch anywhere in this set ---------------------
from . import eval_der, eval_metrics, labels, passthrough, tokenizer  # noqa: F401

__version__ = "1.0.0"

#: submodules that pull in torch (or are heavy) and load on first use
_LAZY_MODULES = ("model", "inference", "pipeline", "eval_report")

#: names re-exported from those submodules
_LAZY_ATTRS = {
    "DiacritizerModel": ("model", "DiacritizerModel"),
    "build_from_config": ("model", "build_from_config"),
    "TashkeelNano": ("inference", "TashkeelNano"),
    "strip_marks": ("inference", "strip_marks"),
    "reorder_shadda_first": ("inference", "reorder_shadda_first"),
    "window_starts": ("inference", "window_starts"),
    "resolve_decode": ("inference", "resolve_decode"),
    "PRESETS": ("inference", "PRESETS"),
    "DECODE_ARMS": ("inference", "DECODE_ARMS"),
    "ORDERS": ("inference", "ORDERS"),
}

__all__ = (list(_LAZY_MODULES) + list(_LAZY_ATTRS)
           + ["eval_der", "eval_metrics", "labels", "passthrough", "tokenizer",
              "__version__"])


def __getattr__(name):
    if name in _LAZY_MODULES:
        mod = importlib.import_module("." + name, __name__)
        globals()[name] = mod
        return mod
    if name in _LAZY_ATTRS:
        mod_name, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module("." + mod_name, __name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(_LAZY_MODULES) | set(_LAZY_ATTRS))
