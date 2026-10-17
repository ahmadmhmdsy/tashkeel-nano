"""R-001 arm A1 / A1-small: LoRA SFT, plain -> diacritized, one epoch, fp16 autocast.

Base weights stay fp32 (Turing has no bf16; fp32 masters avoid fp16 underflow
in the optimizer), the forward/backward runs under fp16 autocast with a
GradScaler. --bs is the micro-batch; --accum micro-batches make one optimizer
step (bs 4 peaks near 3 GB on SmolLM2-135M; bs 16 needs 8.6 GB and the
Windows driver silently spills to system RAM, about 4x slower). --max-minutes cuts the run for timed pilots; the log records
supervised tokens/s, which is what the budget is planned in.

    python r001/sft.py --model smollm2-135m --seed 0 --max-minutes 30 --out runs/r001/pilot_a1_small
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import cap_vram, MODELS, REPO, prompt

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, rows, max_len):
    out = []
    for r in rows:
        p = tok(prompt(r["plain"]), add_special_tokens=False)["input_ids"]
        t = tok(r["diac"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = (p + t)[:max_len]
        labels = ([-100] * len(p) + t)[:max_len]
        out.append((ids, labels))
    return out


def batches(data, bs, rng):
    idx = list(range(len(data)))
    rng.shuffle(idx)
    groups = [idx[i:i + bs * 50] for i in range(0, len(idx), bs * 50)]
    out = []
    for g in groups:  # sort inside a bucket to cut padding, keep global order random
        g.sort(key=lambda i: len(data[i][0]))
        out += [g[i:i + bs] for i in range(0, len(g), bs)]
    rng.shuffle(out)
    return out


def collate(data, b, pad):
    n = max(len(data[i][0]) for i in b)
    ids = torch.full((len(b), n), pad)
    lab = torch.full((len(b), n), -100)
    att = torch.zeros((len(b), n), dtype=torch.long)
    for j, i in enumerate(b):
        x, y = data[i]
        ids[j, :len(x)] = torch.tensor(x)
        lab[j, :len(y)] = torch.tensor(y)
        att[j, :len(x)] = 1
    return ids, lab, att


@torch.no_grad()
def dev_loss(model, data, bs, pad):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(data), bs):
        ids, lab, att = (t.cuda() for t in collate(data, list(range(i, min(i + bs, len(data)))), pad))
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        k = int((lab[:, 1:] != -100).sum())
        tot, n = tot + loss.item() * k, n + k
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=str(REPO / "data" / "r001"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    a = ap.parse_args()
    cap_vram()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    load = lambda f: [json.loads(l) for l in open(Path(a.data) / f, encoding="utf-8")]
    train = encode(tok, load("train.jsonl"), a.max_len)
    dev = encode(tok, load("dev.jsonl")[:200], a.max_len)

    model = AutoModelForCausalLM.from_pretrained(MODELS[a.model], dtype=torch.float32).cuda()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    plan = batches(train, a.bs, rng)
    n_opt = math.ceil(len(plan) / a.accum)
    warm = min(100, n_opt // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, n_opt) / n_opt)))
    scaler = torch.amp.GradScaler("cuda")
    log = open(out / "log.jsonl", "w", encoding="utf-8")
    t0, toks, step = time.time(), 0, 0
    meta = {"model": a.model, "seed": a.seed, "steps_planned": n_opt, "bs": a.bs, "accum": a.accum, "lr": a.lr,
            "rank": a.rank, "train_rows": len(train),
            "train_tokens": sum(len(x) for x, _ in train)}
    print(meta)
    log.write(json.dumps({"meta": meta}) + "\n")
    model.train()
    for i, b in enumerate(plan, 1):
        ids, lab, att = (t.cuda() for t in collate(train, b, pad))
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        scaler.scale(loss / a.accum).backward()
        toks += int(att.sum())
        if i % a.accum and i < len(plan):
            continue
        step += 1
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        el = time.time() - t0
        if step % a.log_every == 0:
            rec = {"step": step, "loss": round(loss.item(), 4), "tok_s": round(toks / el),
                   "elapsed_min": round(el / 60, 2), "scale": scaler.get_scale(),
                   "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2)}
            if step % a.eval_every == 0:
                rec["dev_loss"] = round(dev_loss(model, dev, a.bs, pad), 4)
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if a.max_minutes and el > a.max_minutes * 60:
            break
    el = time.time() - t0
    end = {"end": True, "steps_done": step, "steps_planned": n_opt, "tok_s": round(toks / el),
           "elapsed_min": round(el / 60, 2), "dev_loss": round(dev_loss(model, dev, a.bs, pad), 4),
           "projected_epoch_hours": round(meta["train_tokens"] / (toks / el) / 3600, 2)}
    print(end)
    log.write(json.dumps(end) + "\n")
    model.save_pretrained(out / "adapter")


if __name__ == "__main__":
    main()
