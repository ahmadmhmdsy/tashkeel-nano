'''Per-family model loading and inference.

Each family has one loader and one predictor, so a model's quirks (a bundled
package, a prompt wrapper, a hard token cap) live in exactly one place.
'''
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

CACHE = Path(os.environ.get('TASHKEEL_SPACE_CACHE', Path.home() / '.cache' / 'tashkeel_space'))


def _load_bundled_package(local_dir: str, name: str):
    '''Import a repo's bundled ``tashkeel_nano`` package under a unique name.

    The two published nano repos ship *different* renderers (1p1m emits NFC,
    4p5m ships the pre-NFC renderer), so they must not share one module
    object. Each gets its own module name and its own search path.
    '''
    pkg_dir = os.path.join(local_dir, 'tashkeel_nano')
    spec = importlib.util.spec_from_file_location(
        name,
        os.path.join(pkg_dir, '__init__.py'),
        submodule_search_locations=[pkg_dir],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------- our nano

def load_nano(repo_id: str, device: str = 'cpu', key: str = 'nano'):
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id, allow_patterns=['*.safetensors', '*.json', '*.py', '*.md']
    )
    module = _load_bundled_package(local, 'tashkeel_nano__' + key)
    return module.TashkeelNano.from_pretrained(repo_id, device=device)


def predict_nano(handle, text: str) -> str:
    return handle.diacritize(text)


# -------------------------------------------------------------------- mishkala

def load_mishkala(repo_id: str = 'flokymind/mishkala', device: str = 'cpu', key: str = 'mishkala'):
    '''mishkala ships a custom ``model.py`` plus a CRF decoder.

    ``load_mishkala`` auto-selects CUDA when available, so the device is
    passed explicitly to keep the Space deterministic.
    '''
    from huggingface_hub import hf_hub_download

    deps = CACHE / 'mishkala_deps'
    deps.mkdir(parents=True, exist_ok=True)
    target = deps / 'model.py'
    if not target.exists():
        shutil.copy(hf_hub_download(repo_id, 'model.py'), target)
    if str(deps) not in sys.path:
        sys.path.insert(0, str(deps))
    import model as mishkala_module

    model, tokenizer, resolved = mishkala_module.load_mishkala(device=device)
    return (mishkala_module, model, tokenizer, resolved)


def predict_mishkala(handle, text: str) -> str:
    module, model, tokenizer, device = handle
    return module.tashkeel(text, model, tokenizer, device)


# --------------------------------------------------------------------- seq2seq

def load_seq2seq(repo_id: str, device: str = 'cpu', key: str = 'seq2seq'):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id).to(device).eval()
    return (tokenizer, model, device)


def predict_seq2seq(handle, text: str, max_length: int = 256) -> str:
    '''Greedy seq2seq decoding.

    Greedy rather than beam search: beams=4 measured 71.7 s per sentence on
    CPU versus 12.0 s greedy, for output that differed only in mark order.
    '''
    import torch

    tokenizer, model, device = handle
    inputs = tokenizer(
        text, return_tensors='pt', truncation=True, max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs, max_length=max_length, num_beams=1, do_sample=False
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


LOADERS = {
    'nano': load_nano,
    'mishkala': load_mishkala,
    'seq2seq': load_seq2seq,
}

PREDICTORS = {
    'nano': predict_nano,
    'mishkala': predict_mishkala,
    'seq2seq': predict_seq2seq,
}