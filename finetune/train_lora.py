"""LoRA fine-tuning of a small open-weight model on CPU. `python finetune/train_lora.py`

Base model : HuggingFaceTB/SmolLM2-135M-Instruct (Apache-2.0)
Method     : LoRA (r=16, alpha=32) on all linear projections, loss on the completion tokens only.
"""
import argparse
import json
import platform
import random
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def encode(tok, row, max_len):
    messages = [{"role": "system", "content": row["prompt_system"]}, {"role": "user", "content": row["prompt_user"]}]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(row["target_text"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + t_ids)[:max_len]
    labels = ([-100] * len(p_ids) + t_ids)[:max_len]
    return ids, labels


def collate(batch, pad_id):
    n = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), n), pad_id)
    lab = torch.full((len(batch), n), -100)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for k, (i, l) in enumerate(batch):
        ids[k, : len(i)], lab[k, : len(l)], att[k, : len(i)] = torch.tensor(i), torch.tensor(l), 1
    return ids, att, lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full run; >0 truncates (smoke test)")
    ap.add_argument("--out", default=str(HERE / "adapter"))
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    data = [encode(tok, r, args.max_len) for r in load_rows(HERE / "data" / "train.jsonl")]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_per_epoch = (len(data) + args.batch_size - 1) // args.batch_size
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.05, 1 - s / total))

    log, step, t0 = [], 0, time.time()
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(data)
        for b in range(0, len(data), args.batch_size):
            ids, att, lab = collate(data[b : b + args.batch_size], tok.pad_token_id)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            log.append({"step": step, "epoch": epoch + 1, "loss": round(loss.item(), 4)})
            if step % 5 == 0 or step == 1:
                print(f"epoch {epoch + 1} step {step}/{total} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    model.save_pretrained(args.out)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "training_log.json").write_text(json.dumps({
        "status": "EXECUTED", "base_model": BASE, "args": vars(args), "steps": step,
        "train_seconds": round(time.time() - t0, 1), "loss_curve": log,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "peft": peft.__version__},
        "hardware": f"CPU only ({platform.processor() or platform.machine()}), threads={args.threads}",
    }, indent=1))
    print("saved adapter to", args.out)


if __name__ == "__main__":
    main()
