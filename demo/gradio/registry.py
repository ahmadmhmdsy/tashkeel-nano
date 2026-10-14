'''Model roster plus lazy loading and LRU eviction.

Roster membership is result-based. Each of these models was probed locally
before being admitted (``scratch/probe_one.py``, ``probe_two.py``):

  mishkala        loads, 12.5 M params, correct output      -> included
  flan-t5-small   loads, 77 M params, correct, 12.0 s CPU   -> included, slow tier
  Tashkeel-50M    degenerate repetition loop at every setting -> EXCLUDED
  Fine-Tashkeel   correct but 307 s per sentence on CPU     -> EXCLUDED
  Tashkeel-v4     gated repo, unusable in a public Space    -> EXCLUDED
  Tashkeel-v3     loads only with trust_remote_code=True   -> EXCLUDED
                  (arbitrary remote code execution is not a risk this Space
                   should take on for one extra column)

Excluded models are not listed anywhere in the UI. A model that cannot run
must not appear as a broken button.
'''
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

import adapters


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    repo: str
    params_m: float
    family: str
    tier: str
    mark_order: str
    default_on: bool
    note: str

    @property
    def size_mb(self) -> float:
        return round(self.params_m * 4.0, 1)

    @property
    def is_ours(self) -> bool:
        return self.label.endswith('(ours)')


MODELS: List[ModelSpec] = [
    ModelSpec(
        key='nano1p1m',
        label='tashkeel-nano-1p1m (ours)',
        repo='ahmadsy/tashkeel-nano-1p1m',
        params_m=1.08,
        family='nano',
        tier='fast',
        mark_order='nfc',
        default_on=True,
        note='Centered sliding-window decode. Emits Unicode NFC.',
    ),
    ModelSpec(
        key='nano4p5m',
        label='tashkeel-nano-4p5m (ours)',
        repo='ahmadsy/tashkeel-nano-4p5m',
        params_m=4.49,
        family='nano',
        tier='fast',
        mark_order='legacy',
        default_on=True,
        note='Superseded by 1p1m. Ships the pre-NFC (shadda-first) renderer.',
    ),
    ModelSpec(
        key='mishkala',
        label='mishkala (flokymind)',
        repo='flokymind/mishkala',
        params_m=12.5,
        family='mishkala',
        tier='fast',
        mark_order='nfc',
        default_on=True,
        note='Mamba + Transformer with a CRF decoder. Downloads model.py.',
    ),
    ModelSpec(
        key='flan',
        label='arabic-tashkeel-flan-t5-small',
        repo='Abdou/arabic-tashkeel-flan-t5-small',
        params_m=77.0,
        family='seq2seq',
        tier='slow',
        mark_order='legacy',
        default_on=False,
        note='Greedy decoding; ~12 s per sentence on a local CPU, slower on a free Space.',
    ),
]

BY_KEY: Dict[str, ModelSpec] = {m.key: m for m in MODELS}


class Registry:
    '''Lazy model loading with LRU eviction and a repeat-result cache.'''

    def __init__(self, specs: List[ModelSpec] = None, max_resident: int = 3):
        self.specs = list(specs if specs is not None else MODELS)
        self.by_key = {m.key: m for m in self.specs}
        self.max_resident = max_resident
        self._resident: 'OrderedDict[str, object]' = OrderedDict()
        self._results: Dict[str, str] = {}
        self._lock = threading.RLock()

    # -------------------------------------------------------------- handles
    def _evict(self) -> None:
        while len(self._resident) > self.max_resident:
            key, _ = self._resident.popitem(last=False)
            # Dropping the handle is enough; the loader owns no global state
            # except mishkala's imported module, which is harmless to keep.
    def get(self, key: str, device: str = 'cpu'):
        spec = self.by_key[key]
        with self._lock:
            if key in self._resident:
                self._resident.move_to_end(key)
                return self._resident[key]
            loader = adapters.LOADERS[spec.family]
            handle = loader(spec.repo, device=device, key=key)
            self._resident[key] = handle
            self._evict()
            return handle

    def resident(self) -> List[str]:
        with self._lock:
            return list(self._resident.keys())

    def unload(self, key: str) -> None:
        with self._lock:
            self._resident.pop(key, None)

    # ------------------------------------------------------------ inference
    def predict(self, key: str, text: str, device: str = 'cpu') -> Dict:
        '''Diacritize ``text`` with one model. Never raises; errors are returned.'''
        spec = self.by_key[key]
        digest = hashlib.sha1(('%s\x00%s' % (key, text)).encode('utf-8')).hexdigest()
        with self._lock:
            if digest in self._results:
                return {
                    'key': key, 'label': spec.label, 'out': self._results[digest],
                    'seconds': 0.0, 'cached': True, 'error': None,
                }
        started = time.time()
        try:
            handle = self.get(key, device)
            predict = adapters.PREDICTORS[spec.family]
            out = predict(handle, text)
            error: Optional[str] = None
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            out = text
            error = '%s: %s' % (type(exc).__name__, exc)
        seconds = time.time() - started
        if error is None:
            with self._lock:
                if len(self._results) > 4096:
                    self._results.clear()
                self._results[digest] = out
        return {
            'key': key, 'label': spec.label, 'out': out,
            'seconds': seconds, 'cached': False, 'error': error,
        }


REGISTRY = Registry()


def resolve(keys: Optional[List[str]] = None) -> List[ModelSpec]:
    '''Return the specs for ``keys``, defaulting to the default-on roster.'''
    if not keys:
        return [m for m in MODELS if m.default_on]
    unknown = [k for k in keys if k not in BY_KEY]
    if unknown:
        raise KeyError('unknown model key(s): %s' % ', '.join(unknown))
    return [BY_KEY[k] for k in keys]