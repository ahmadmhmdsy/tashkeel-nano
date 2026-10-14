# Training

## The two-stage shape of the line

Every model here starts from the **same stage-1 causal character LM**, which is
trained on raw text with no diacritic labels at all. Its weights are then reused
as the encoder initialisation of a diacritization run. This is a warm start, not
a distillation: the LM head is tied to the embedding and simply is not used
afterwards.

    stage1lm -> e63s_s4_12l160                                   (nano4p5m)
    stage1lm -> e67_p1_lm64 -> e67_c1_ft64_fullv3
             -> e69_c4_ctx256 -> e98_soupB                       (nano1p1m)

`configs/lineage/` holds the recorded config for each step, and
`configs/nano1p1m.yaml` / `configs/nano4p5m.yaml` hold the final ones.
`configs/lineage/e70_c4b_ctx256.yaml` is a **sibling branch**, not part of the
1p1m path — it is the run the older "deployed gold" soup was built from, kept
because the decode experiments are recorded against it.

## Stage 1 — the causal character LM

    python scripts/prep_rawlm.py
    python scripts/train_lm.py --config configs/stage1_lm.yaml

`scripts/prep_rawlm.py` strips marks from the raw corpora and writes
`data/stage1/tokens/{train_ids,val_ids}.npy` at ctx 512, with a contamination
guard: any window whose mark-stripped text contains a 40-character gate shingle,
or that equals a gate row exactly, is excluded. Val is every 100th distinct
document. The recorded run kept 1,226,329 train windows and 12,580 val windows,
with per-source character budgets so no single corpus dominates the class mix.

The LM itself is the same `DiacritizerModel` built with `causal: true` and a
tied head, so it introduces no extra parameters.

## Stage 2 — diacritization

    python scripts/train.py --config configs/nano4p5m.yaml
    python scripts/train.py --config configs/nano1p1m.yaml

`scripts/train.py` carries the discipline contract that produced the published
models:

- **fp16 autocast only.** Turing (sm_75) has no bf16 and no flash-attention, so
  the model uses PyTorch SDPA.
- **Zero-flag auto-resume.** After any crash or kill, re-run the *exact* original
  command with *zero* extra flags. It finds the newest
  `runs/<phase>/checkpoint-*` (model + optimizer + step) and continues.
  There is no resume flag to remember and no way to accidentally restart from
  scratch.
- **Checkpoint rotation** with `save_total_limit=3`, so a long run cannot fill
  the disk.
- **Single-GPU exclusivity.** Never launch a second `train.py` while one is
  running.
- **An always-on monitor** (`tashkeel_nano/pipeline/monitor.py`) writes
  `runs/<phase>/monitor/log.jsonl` and `monitor/summary.json`: one record per
  step, eval, save, VRAM probe and anomaly. Three consecutive non-finite train
  losses, or any non-finite eval metric, raises a `DIVERGED` flag. The monitor
  never raises into the training loop.

### What the runs actually were

| | nano4p5m | nano1p1m |
|---|---|---|
| phase | `e63s_s4_12l160` | `e98_soupB` |
| init | `runs/stage1lm/final/model.pt` | `runs/e69_c4_ctx256/final/model.pt` |
| bed | `v3` (ctx 128) | `v3full` (ctx 256) |
| steps | 20,000 | 60,000 |
| batch | 32 | 64 |
| LR | 2e-4 constant (AdamW), weight decay 0.1 | same |
| seed | 777 | 20261012 |
| published artifact | step **19,500** (best val loss 0.17575) | uniform soup of the 12 probes at 32,500 … 60,000 |

The 1.1 M model's published weights are **not** the final step. Continuing
training from 30 k to 60 k produced no monotone gain — the individual probes span
12.88 – 14.17 on the registered gate with no trend — but their **uniform average**
is 11.51, which is better than every single member and better than the 30 k
anchor. Averaging over a wide, same-basin checkpoint set is the mechanism, and
`scripts/soup.py` reproduces it exactly:

    python scripts/soup.py --run runs/e98_soupB --members 32500:60000:2500 \
        --out runs/e98_soupB/soupB_uniform

It averages every tensor in fp32, casts back to the members' dtype, saves the
same `{model, step, config}` wrapper shape the loaders expect, and writes
`soup.json` with each member's path, sha256, and relative L2 distance to the
result — the basin check the soup experiments used. CPU only.

## Gate probes

A run configured with `gate_eval_every` saves weight snapshots under
`runs/<phase>/gate_probe/`. That is where the 12 soup members come from, and
where `e67_c1_ft64_fullv3/gate_probe/best_gate_weights.pt` — the init of
`e69_c4_ctx256` — came from.

## Hardware

The published models were trained on a single Quadro RTX 3000 (6 GB). The 1.1 M
model is far below that ceiling; the 4.5 M model fits at batch 32 / ctx 128. The
recorded configs carry a `vram_cap_gib` for the LM run because WDDM can spill
silently when a training job exceeds the card's residency budget.

## Reproducibility

`scripts/train.py` seeds from the config (`seed`) and draws batches with a fixed
generator, so a given config and bed produce the same trajectory. The bed itself
is deterministic: `tokenize_data.py` shuffles with a literal seed and
`e97_build_bed.py` reproduces that permutation by construction rather than by
argument. Expected row counts are listed in [reproduction.md](reproduction.md).
