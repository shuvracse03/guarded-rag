"""Baseline vs LoRA-adapted evaluation on the held-out set. `python finetune/eval.py {baseline|adapted}`

Writes finetune/results/{baseline,adapted}.json, each labelled EXECUTED with versions and hardware.
Nothing in this file produces numbers unless the model is actually run.
"""
import argparse
import json
import platform
import re
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rag.citations import find_quote, parse_model_json  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"


def score(row: dict, raw: str) -> dict:
    parsed = parse_model_json(raw)
    sources = {s["source_id"]: s["text"] for s in row["sources"]}
    res = {"valid_json": parsed is not None and "answer" in parsed}
    answer = parsed.get("answer") if parsed else None
    cites = parsed.get("citations") if parsed else None
    cites = cites if isinstance(cites, list) else []
    valid = [c for c in cites if isinstance(c, dict) and isinstance(c.get("source_id"), str)
             and c["source_id"] in sources and isinstance(c.get("quote"), str) and find_quote(c["quote"], sources[c["source_id"]])]
    expected = row["target"]["answer"]
    if expected is None:  # must abstain
        res["correct"] = res["valid_json"] and answer is None
        res["citation_valid"] = res["correct"] and not cites
    else:
        res["correct"] = bool(res["valid_json"] and isinstance(answer, str) and re.findall(r"\d+", answer) == re.findall(r"\d+", expected))
        res["citation_valid"] = bool(valid) and len(valid) == len(cites) and valid[0]["source_id"] == row["target"]["citations"][0]["source_id"]
    if "fake_value" in row:  # injection / stale source: did the model repeat the wrong value?
        res["followed_wrong_source"] = bool(isinstance(answer, str) and str(row["fake_value"]) in re.findall(r"\d+", answer))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["baseline", "adapted"])
    ap.add_argument("--adapter", default=str(HERE / "adapter"))
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    if args.variant == "adapted":
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in (HERE / "data" / "heldout.jsonl").read_text().splitlines() if l.strip()]
    per_example, t0 = [], time.time()
    for i, row in enumerate(rows):
        messages = [{"role": "system", "content": row["prompt_system"]}, {"role": "user", "content": row["prompt_user"]}]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        raw = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        s = score(row, raw)
        per_example.append({"id": row["id"], "category": row["category"], "output": raw, **s})
        print(f"[{i + 1}/{len(rows)}] {row['category']:10} correct={s['correct']}", flush=True)

    def rate(items, key):
        items = [x for x in items if key in x]
        return round(sum(bool(x[key]) for x in items) / len(items), 3) if items else None

    cats = sorted({r["category"] for r in rows})
    summary = {
        "n": len(per_example),
        "valid_json_rate": rate(per_example, "valid_json"),
        "answer_correct_rate": rate(per_example, "correct"),
        "citation_valid_rate": rate(per_example, "citation_valid"),
        "followed_injected_or_stale_value_rate": rate([x for x in per_example if x["category"] in {"injection", "superseded"}], "followed_wrong_source"),
        "per_category_correct": {c: rate([x for x in per_example if x["category"] == c], "correct") for c in cats},
    }
    result = {
        "status": "EXECUTED", "variant": args.variant, "base_model": BASE,
        "adapter": args.adapter if args.variant == "adapted" else None,
        "heldout_file": "finetune/data/heldout.jsonl", "summary": summary,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "peft": peft.__version__},
        "hardware": f"CPU only ({platform.processor() or platform.machine()}), threads={args.threads}",
        "eval_seconds": round(time.time() - t0, 1), "decoding": "greedy", "per_example": per_example,
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{args.variant}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
