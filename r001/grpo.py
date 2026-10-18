"""R-001 arm A2 / A2-small: GRPO from the same-seed A1 adapter, reward = 1 - DER.

Per step, P prompts get G sampled completions each (temperature 1). The
advantage is the reward minus its group mean, divided by the group std; groups
whose samples all score the same carry no signal and are skipped. One policy
update per batch of samples (fully on-policy), so the PPO ratio is 1 and the
loss reduces to -mean(advantage * token log-prob), with no clip and no KL term.
The nano_SLMs RLCD code the pre-registration names was not on disk on
2026-10-18, so this is written fresh.

The scored adapter is the one with the best dev DER (adapter/); the last one is
kept as adapter_last/. Every --eval-every steps, greedy DER on the 500-sentence dev slice (Fadel
val.txt chunks, data/r001/dev.jsonl). Stop rule (pre-registered): mean reward
rises while dev DER fails to improve for 3 consecutive evaluations.

    python r001/grpo.py --model smollm2-135m --seed 0 --init runs/r001/a1_small_s0/adapter \
        --out runs/r001/a2_small_s0 --max-minutes 30
"""
from __future__ import annotations

import argparse
import json
import random
import time
import unicodedata
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import cap_vram, MODELS, REPO, prompt, reward
from tashkeel_nano import eval_der


def first_line(text):
    return unicodedata.normalize("NFC", text.split("\n")[0].strip())


@torch.no_grad()
def sample(model, tok, plains, g, temp):
    """G sampled completions per plain prompt -> (sequences, prompt_len, texts)."""
    tok.padding_side = "left"
    enc = tok([prompt(p) for p in plains], return_tensors="pt", padding=True,
              add_special_tokens=False).to("cuda")
    cap = int(enc["input_ids"].shape[1] * 3.0) + 16
    with torch.autocast("cuda", dtype=torch.float16):
        gen = model.generate(**enc, max_new_tokens=cap, do_sample=True, temperature=temp, top_k=0, top_p=1.0,
                             num_return_sequences=g, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    plen = enc["input_ids"].shape[1]
    texts = [first_line(tok.decode(r, skip_special_tokens=True)) for r in gen[:, plen:]]
    return gen, enc["attention_mask"].repeat_interleave(g, 0), plen, texts


def completion_mask(seq, plen, eos, pad):
    """1 on generated tokens up to and including the first EOS."""
    comp = seq[:, plen:]
    is_end = comp == eos
    after = (is_end.cumsum(1) - is_end.long()) > 0  # strictly after the first EOS
    m = ~after
    if pad != eos:
        m &= comp != pad
    return m


def policy_backward(model, scaler, seq, prompt_att, plen, mask, adv, micro):
    """-mean(adv * mean token log-prob), backpropagated in micro-batches of sequences."""
    n, total = len(seq), 0.0
    for i in range(0, n, micro):
        m = mask[i:i + micro]
        L = max(int(m.sum(1).max()), 1)  # drop the padding past the longest completion
        s, m = seq[i:i + micro, :plen + L], m[:, :L]
        att = torch.cat([prompt_att[i:i + micro], m.long()], 1)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=s, attention_mask=att).logits[:, plen - 1:-1].float()
        lp = logits.gather(-1, s[:, plen:, None]).squeeze(-1) - torch.logsumexp(logits, -1)
        mf = m.float()
        per_seq = (lp * mf).sum(1) / mf.sum(1).clamp(min=1)
        loss = -(adv[i:i + micro] * per_seq).sum() / n
        scaler.scale(loss).backward()
        total += loss.item()
    return total


@torch.no_grad()
def dev_der(model, tok, dev, bs):
    model.eval()
    tok.padding_side = "left"
    der, words = 0.0, 0
    for i in range(0, len(dev), bs):
        group = dev[i:i + bs]
        enc = tok([prompt(r["plain"]) for r in group], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            gen = model.generate(**enc, max_new_tokens=int(enc["input_ids"].shape[1] * 3.0) + 16, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for r, row in zip(group, gen[:, enc["input_ids"].shape[1]:]):
            m = eval_der.score_line(first_line(tok.decode(row, skip_special_tokens=True)), [r["diac"]])
            der += m["DER"] * m["words"]
            words += m["words"]
    model.train()
    return der / max(words, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", required=True, help="the same-seed A1 adapter")
    ap.add_argument("--data", default=str(REPO / "data" / "r001"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", type=int, default=5000)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--per-step", type=int, default=2, help="prompts per optimizer step")
    ap.add_argument("--micro", type=int, default=2, help="sequences per backward pass")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--dev-bs", type=int, default=32)
    a = ap.parse_args()
    cap_vram()

    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    load = lambda f: [json.loads(l) for l in open(Path(a.data) / f, encoding="utf-8")]
    # Prompts: seed-0 draw from A1's subset (same for every A2 seed); order shuffled per seed.
    rows = random.Random(0).sample(load("train.jsonl"), a.prompts)
    random.Random(a.seed).shuffle(rows)
    dev = load("dev.jsonl")

    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODELS[a.model], dtype=torch.float32).cuda()
    model = PeftModel.from_pretrained(base, a.init, is_trainable=True)
    model.config.use_cache = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")

    log = open(out / "log.jsonl", "w", encoding="utf-8")
    n_steps = len(rows) // a.per_step
    meta = {"model": a.model, "seed": a.seed, "init": a.init, "prompts": len(rows), "group": a.group,
            "per_step": a.per_step, "steps_planned": n_steps, "lr": a.lr, "temp": a.temp}
    print(meta, flush=True)
    log.write(json.dumps({"meta": meta}) + "\n")

    def emit(rec):
        print(rec, flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    best = dev_der(model, tok, dev, a.dev_bs)
    emit({"step": 0, "dev_DER": round(best, 5)})
    model.save_pretrained(out / "adapter")
    t0, step, stale, stop = time.time(), 0, 0, None
    win_r, win_zero, win_skip, gen_tok, prev_r = [], [], 0, 0, None
    model.train()
    for step in range(1, n_steps + 1):
        batch = rows[(step - 1) * a.per_step: step * a.per_step]
        model.eval()
        seq, patt, plen, texts = sample(model, tok, [r["plain"] for r in batch], a.group, a.temp)
        model.train()
        torch.cuda.empty_cache()  # release the sampling KV cache before the backward pass
        r = torch.tensor([reward(t, batch[k // a.group]["plain"], [batch[k // a.group]["diac"]])
                          for k, t in enumerate(texts)], device="cuda").view(len(batch), a.group)
        win_r += r.flatten().tolist()
        win_zero += [float(x == 0.0) for x in r.flatten().tolist()]
        std = r.std(1, keepdim=True)
        adv = ((r - r.mean(1, keepdim=True)) / (std + 1e-4)).flatten()
        keep = (std > 1e-6).expand(-1, a.group).flatten()
        mask = completion_mask(seq, plen, tok.eos_token_id, tok.pad_token_id)
        gen_tok += int(mask.sum())
        if keep.any():
            policy_backward(model, scaler, seq[keep], patt[keep], plen, mask[keep], adv[keep], a.micro)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        else:
            win_skip += 1
        el = time.time() - t0
        if step % a.log_every == 0:
            emit({"step": step, "reward": round(sum(win_r) / len(win_r), 4),
                  "zero_frac": round(sum(win_zero) / len(win_zero), 4), "skipped": win_skip,
                  "gen_tok_s": round(gen_tok / el), "elapsed_min": round(el / 60, 2),
                  "scale": scaler.get_scale(), "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                  "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2)})
        if step % a.eval_every == 0:
            mean_r = sum(win_r) / len(win_r)
            d = dev_der(model, tok, dev, a.dev_bs)
            improved = d < best
            if improved:
                best = d
                model.save_pretrained(out / "adapter")
            # consecutive: an improvement, or a non-rising reward, resets the count
            stale = stale + 1 if not improved and prev_r is not None and mean_r > prev_r else 0
            prev_r = mean_r
            emit({"step": step, "dev_DER": round(d, 5), "best_dev_DER": round(best, 5),
                  "window_reward": round(mean_r, 4), "stale": stale})
            win_r, win_zero, win_skip = [], [], 0
            if stale >= 3:
                stop = "reward rose while dev DER did not improve for 3 evaluations"
                break
        if a.max_minutes and el > a.max_minutes * 60:
            stop = "max-minutes"
            break
    model.save_pretrained(out / "adapter_last")
    el = time.time() - t0
    emit({"end": True, "steps_done": step, "steps_planned": n_steps, "elapsed_min": round(el / 60, 2),
          "best_dev_DER": round(best, 5), "stop": stop,
          "projected_hours": round(el / step * n_steps / 3600, 2)})


if __name__ == "__main__":
    main()
