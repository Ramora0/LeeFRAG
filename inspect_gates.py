"""Load a Q-Former checkpoint and print the learnable residual gate values."""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect learnable gate values in a Q-Former checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    # The checkpoint may store the state dict directly or under a key
    if "qformer_state_dict" in ckpt:
        state_dict = ckpt["qformer_state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    gate_keys = sorted(k for k in state_dict if "gate" in k)

    if not gate_keys:
        print("No gate parameters found in checkpoint.")
        print(f"Available keys ({len(state_dict)}):")
        for k in sorted(state_dict):
            print(f"  {k}")
        return

    print(f"Found {len(gate_keys)} gate(s):\n")
    print(f"{'Layer':<50} {'Raw':>8} {'Sigmoid':>8}")
    print("-" * 70)
    for key in gate_keys:
        raw = state_dict[key].item()
        sig = torch.sigmoid(state_dict[key]).item()
        print(f"{key:<50} {raw:>8.4f} {sig:>8.4f}")

    # Summary
    raw_vals = [state_dict[k].item() for k in gate_keys]
    sig_vals = [torch.sigmoid(state_dict[k]).item() for k in gate_keys]
    print(f"\nSigmoid range: [{min(sig_vals):.4f}, {max(sig_vals):.4f}]")
    print(f"Sigmoid mean:  {sum(sig_vals) / len(sig_vals):.4f}")
    print(f"\nNote: sigmoid near 0 = identity/no-op, near 1 = full effect")


if __name__ == "__main__":
    main()
