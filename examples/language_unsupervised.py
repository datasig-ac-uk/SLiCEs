"""
Train a stacked SLiCE language model and export hidden states.

Flags:
  --horizon {single,multi}     Prediction horizon(s)
  --export-mode {all,final}    Export H_{T-1} only or all H_t

Outputs:
  <export-path>.dat  : numpy memmap of hidden states
  <export-path>.npz  : metadata (shape, dtype, labels)
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from slices import StackedSLiCE

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
MAX_CHAR_VOCAB = 256
MULTI_HORIZONS = (1, 2, 4)


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SLiCE and export hidden states")

    # Data
    parser.add_argument("--n-per-lang", type=int, default=8000)
    parser.add_argument("--max-seq-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--cache-dir", type=str, default="/tmp/wiki_cache")
    parser.add_argument("--processed-cache", type=str, default="processed_dataset.pt")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )

    # Model
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=192)

    # Experimental flags
    parser.add_argument("--horizon", choices=["single", "multi"], default="single")
    parser.add_argument("--export-mode", choices=["all", "final"], default="all")

    # Output
    parser.add_argument("--export-path", type=str, default="slice_embeddings")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


# ---------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------

def collect_wikipedia_texts(
    config: str, *, needed: int, seed: int, cache_dir: str
) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(
        "wikimedia/wikipedia",
        config,
        split="train",
        cache_dir=cache_dir,
    ).shuffle(seed=seed)

    texts: list[str] = []
    for row in ds:
        t = normalise(row.get("text", ""))
        if len(t) >= 8:
            texts.append(t)
        if len(texts) >= needed:
            break

    return texts


def build_vocab(texts: list[str]) -> dict[str, int]:
    counter = Counter("".join(texts))
    vocab = list(SPECIAL_TOKENS)
    remaining = MAX_CHAR_VOCAB - len(vocab)

    common = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:remaining]
    vocab.extend(ch for ch, _ in common)

    return {tok: i for i, tok in enumerate(vocab)}


def encode_texts(
    texts: list[str], vocab: dict[str, int], max_len: int
) -> torch.Tensor:
    pad, bos, eos, unk = (
        vocab["<pad>"],
        vocab["<bos>"],
        vocab["<eos>"],
        vocab["<unk>"],
    )

    rows = []
    for text in texts:
        ids = [bos] + [vocab.get(c, unk) for c in text] + [eos]
        if len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = eos
        else:
            ids.extend([pad] * (max_len - len(ids)))
        rows.append(ids)

    return torch.tensor(rows, dtype=torch.long)


# ---------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------

class SelfSupervisedSLiCE(nn.Module):
    """
    Thin wrapper around StackedSLiCE providing a horizon-aware loss.
    """

    def __init__(self, vocab_size: int, args: argparse.Namespace, pad_id: int):
        super().__init__()
        self.pad_id = pad_id

        if args.horizon == "single":
            label_dim = vocab_size
        else:
            # Predict x[t+1], x[t+2], x[t+4]
            label_dim = (vocab_size, vocab_size, vocab_size)

        self.model = StackedSLiCE(
            num_layers=args.num_layers,
            data_dim=vocab_size,
            hidden_dim=args.hidden_dim,
            label_dim=label_dim,
            tokens=True,
            block_size=args.block_size,
            dropout_rate=args.dropout,
            chunk_size=args.chunk_size,
        )

    def forward(self, x: torch.Tensor):
        """
        Returns:
          - Tensor[B, T, V]              (single horizon)
          - list[Tensor[B, T, V]]        (multi-horizon)
        """
        return self.model(x)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(x)
        targets = x[:, 1:]

        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
            horizons = (1,)
        else:
            horizons = MULTI_HORIZONS

        total = 0.0
        for out, k in zip(outputs, horizons, strict=True):
            # Align predictions and targets
            readout = out[:, :-k]
            tgt = targets[:, k - 1 :]
            mask = tgt.ne(self.pad_id)

            loss = F.cross_entropy(
                readout.reshape(-1, readout.size(-1)),
                tgt.reshape(-1),
                reduction="none",
            ).view_as(tgt)

            total += (loss * mask).sum() / mask.sum().clamp_min(1)

        return total / len(outputs)


# ---------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------

def extract_final(
    states: torch.Tensor, x: torch.Tensor, pad_id: int
) -> torch.Tensor:
    """
    Extract the last valid hidden state H_{T-1} for each sequence.
    """
    targets = x[:, 1:]
    valid = targets.ne(pad_id)
    idx = valid.long().sum(dim=1).clamp_min(1) - 1

    gather = idx.view(-1, 1, 1).expand(-1, 1, states.size(-1))
    return states.gather(1, gather).squeeze(1)


def export_states_memmap(
    model: SelfSupervisedSLiCE,
    x_all: torch.Tensor,
    labels: torch.Tensor,
    *,
    pad_id: int,
    export_mode: str,
    export_path: str,
    hidden_dim: int,
    device: torch.device,
    batch_size: int = 128,
    dtype: str = "float32",
) -> None:
    """
    Memory-safe export of hidden states H_t using numpy memmap.
    """
    assert export_mode in {"final", "all"}
    assert dtype in {"float32", "float16"}

    N, T = x_all.shape
    D = hidden_dim

    shape = (N, D) if export_mode == "final" else (N, T, D)

    mmap_path = export_path + ".dat"
    meta_path = export_path + ".npz"

    print(f"Creating memmap {mmap_path} with shape {shape}")

    mmap = np.memmap(
        mmap_path,
        dtype=dtype,
        mode="w+",
        shape=shape,
    )

    write_idx = 0
    model.eval()
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    with torch.inference_mode():
        for start in range(0, N, batch_size):
            xb = x_all[start : start + batch_size].to(device)

            h = model.model.hidden(xb)  # (B, T, H)

            if export_mode == "final":
                out = extract_final(h, xb, pad_id)
            else:
                out = h

            out = out.to(dtype=torch_dtype).cpu().numpy()
            B = out.shape[0]

            mmap[write_idx : write_idx + B] = out
            write_idx += B

            del xb, h, out
            torch.cuda.empty_cache()

            if write_idx % (batch_size * 10) == 0:
                print(f"  wrote {write_idx}/{N}")

    mmap.flush()

    np.savez(
        meta_path,
        shape=shape,
        dtype=dtype,
        labels=labels.numpy(),
        export_mode=export_mode,
        hidden_dim=D,
    )

    print("Export complete:")
    print(f"  data: {mmap_path}")
    print(f"  meta: {meta_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    processed_cache = Path(args.processed_cache)

    if processed_cache.exists():
        data = torch.load(processed_cache)
        x_all = data["x"]
        labels = data["labels"]
        vocab = data["vocab"]
    else:
        en = collect_wikipedia_texts(
            "20231101.en",
            needed=args.n_per_lang,
            seed=args.seed,
            cache_dir=str(cache_dir),
        )
        fr = collect_wikipedia_texts(
            "20231101.fr",
            needed=args.n_per_lang,
            seed=args.seed + 1,
            cache_dir=str(cache_dir),
        )

        texts = en + fr
        labels = torch.tensor([0] * len(en) + [1] * len(fr))
        perm = torch.randperm(len(texts))
        texts = [texts[i] for i in perm.tolist()]
        labels = labels[perm]

        vocab = build_vocab(texts)
        x_all = encode_texts(texts, vocab, args.max_seq_len)

        torch.save(
            {"x": x_all, "labels": labels, "vocab": vocab},
            processed_cache,
        )

    pad_id = vocab["<pad>"]

    model = SelfSupervisedSLiCE(len(vocab), args, pad_id).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, x_all.size(0), (args.batch_size,))
        xb = x_all[idx].to(device)

        loss = model.loss(xb)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        if step == 1 or step % 100 == 0:
            print(f"step {step:05d} | loss={loss.item():.4f}")

    export_states_memmap(
        model=model,
        x_all=x_all,
        labels=labels,
        pad_id=pad_id,
        export_mode=args.export_mode,
        export_path=args.export_path,
        hidden_dim=args.hidden_dim,
        device=device,
    )


if __name__ == "__main__":
    main()
