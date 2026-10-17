"""Shared R-001 pieces: model ids, the prompt format, and the reward."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tashkeel_nano import eval_der  # noqa: E402

MODELS = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
}
# Plain prompt; no chat template, so the 135M base model and the instruct
# model see the same bytes.
SEP = "\n=\n"
PLAIN_CHUNK = 110  # plain chars per inference chunk ~ the 200-diac-char training chunks


VRAM_CAP_GB = 4.5  # of 6 GB; the desktop holds ~0.6 GB


def cap_vram(gb=VRAM_CAP_GB):
    """Hard-cap this process's CUDA allocator.

    Batches vary in length, so the caching allocator keeps reserving new block
    sizes. On Windows, once reservations pass physical VRAM the driver spills to
    shared system RAM (no OOM), and training slows sharply: 3.1 GB allocated
    became 5.5 GB dedicated + 3.9 GB shared. At the cap the allocator frees its
    cache and retries instead of growing.
    """
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2**30 / total), 0)


def prompt(plain):
    return plain + SEP


def reward(pred, plain, refs):
    """1 - DER, or 0 when the output's undiacritized letters differ from the input."""
    if eval_der.base_of(pred).split() != plain.split():
        return 0.0
    return 1.0 - eval_der.score_line(pred, refs)["DER"]
