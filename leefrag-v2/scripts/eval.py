"""Evaluate a trained checkpoint: CE/ppl at each keep-rate vs the full-context
baseline, plus the empirical keep-rate. (EM/F1 via generation is a TODO.)

  python scripts/eval.py --checkpoint outputs_v2/learned/checkpoint-XXXX/checkpoint.pt \
      --pis 0.5 0.25 0.125
"""

import _bootstrap  # noqa: F401

import argparse
import math

import torch

from leefrag_v2.config import SelectorConfig, V2ModelConfig, V2TrainingConfig
from leefrag_v2.data.adapter import build_blocks, make_dataloaders
from leefrag_v2.loader import build_training_model
from leefrag_v2.model.patch import new_context, set_context
from leefrag_v2.training.losses import ce_on_answer


def load_checkpoint(model, selector, path, device):
    from peft import set_peft_model_state_dict

    ck = torch.load(path, map_location=device)
    set_peft_model_state_dict(model, ck["lora_state_dict"])
    named = dict(model.named_parameters())
    for n, v in ck.get("norm_state_dict", {}).items():
        if n in named:
            named[n].data.copy_(v.to(device))
    if selector is not None and "selector_state_dict" in ck:
        selector.load_state_dict(ck["selector_state_dict"])


@torch.no_grad()
def eval_ce(model, selector, loader, device, mc, *, pi=None, baseline=False,
            use_flex=True, max_batches=0):
    model.eval()
    if selector is not None:
        selector.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        blocks = build_blocks(batch, device)
        if blocks is None:
            continue
        if baseline:
            ctx = new_context(blocks["block_lengths"], blocks["qa_len"], device,
                              mc.num_layers, use_flex=use_flex, teacher_mode=True)
        else:
            ctx = new_context(blocks["block_lengths"], blocks["qa_len"], device,
                              mc.num_layers, use_flex=use_flex, selector=selector, eval_pi=pi)
        set_context(model, ctx)
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            out = model(input_ids=blocks["input_ids"], use_cache=False)
            ce = ce_on_answer(out.logits, blocks["labels"])
        set_context(model, None)
        total += ce.item()
        n += 1
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="rag_v1")
    ap.add_argument("--pis", type=float, nargs="+", default=[0.5, 0.25, 0.125])
    ap.add_argument("--max_eval_batches", type=int, default=0)
    ap.add_argument("--no_flex", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mc, sc = V2ModelConfig(), SelectorConfig()
    cfg = V2TrainingConfig(mode="learned", dataset_name=args.dataset, use_wandb=False)
    model, selector, tokenizer = build_training_model(mc, sc, cfg, device)
    load_checkpoint(model, selector, args.checkpoint, device)
    _, eval_loader = make_dataloaders(tokenizer, mc, cfg)

    use_flex = not args.no_flex
    base = eval_ce(model, selector, eval_loader, device, mc, baseline=True,
                   use_flex=use_flex, max_batches=args.max_eval_batches)
    print(f"{'keep_rate':>10} {'CE':>8} {'ppl':>8} {'dCE_vs_base':>12}")
    print(f"{'full(1.0)':>10} {base:8.4f} {math.exp(min(base,20)):8.2f} {0.0:12.4f}")
    for pi in args.pis:
        ce = eval_ce(model, selector, eval_loader, device, mc, pi=pi,
                     use_flex=use_flex, max_batches=args.max_eval_batches)
        print(f"{pi:>10.3f} {ce:8.4f} {math.exp(min(ce,20)):8.2f} {ce-base:12.4f}")


if __name__ == "__main__":
    main()
