"""
Train a StackedSLiCE to classify sentences as English or French.

This script is a step-by-step tutorial for a simple character-level
classification task:
1. Stream text from Wikipedia in English and French.
2. Build a character vocabulary from training data.
3. Encode each text into fixed-length token IDs.
4. Train a StackedSLiCE encoder and pool sequence states to make prediction.
5. Evaluate validation accuracy and print sample predictions.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter

import torch
import torch.nn.functional as F

from slices import StackedSLiCE

LANGUAGES = ("english", "french")
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")

# Compact defaults for a tutorial that converges quickly.
HIDDEN_DIM = 96
BLOCK_SIZE = 4
NUM_LAYERS = 4
DROPOUT = 0.05
MAX_CHAR_VOCAB = 256
GRAD_CLIP = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tutorial: train StackedSLiCE for character-level "
            "English-vs-French classification."
        )
    )
    parser.add_argument("--train-size", type=int, default=60000)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--max-seq-len", type=int, default=192)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--a-penalty", type=float, default=1e-3)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Keep sampling and weight initialisation reproducible."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalise(text: str) -> str:
    """Lowercase and collapse repeated whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def collect_wikipedia_texts(
    config_name: str,
    *,
    needed: int,
    seed: int,
    cache_dir: str | None,
) -> list[str]:
    """Stream rows from one Wikipedia language split until enough examples."""
    from datasets import load_dataset

    # Streaming avoids loading full Wikipedia dumps in memory.
    stream = load_dataset(
        "wikimedia/wikipedia",
        config_name,
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    ).shuffle(seed=seed, buffer_size=10_000)

    texts: list[str] = []
    for row in stream:
        text = normalise(row.get("text", ""))
        if len(text) < 8:
            continue
        texts.append(text)
        if len(texts) >= needed:
            break

    if len(texts) < needed:
        raise RuntimeError(
            f"Not enough rows for {config_name}. Needed {needed}, got {len(texts)}"
        )
    return texts


def load_wikipedia_en_fr(
    train_size: int,
    val_size: int,
    seed: int,
    cache_dir: str | None,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Build balanced train/validation sets for English (0) and French (1)."""
    train_en = train_size // 2
    train_fr = train_size - train_en
    val_en = val_size // 2
    val_fr = val_size - val_en

    en_texts = collect_wikipedia_texts(
        "20231101.en",
        needed=train_en + val_en,
        seed=seed,
        cache_dir=cache_dir,
    )
    fr_texts = collect_wikipedia_texts(
        "20231101.fr",
        needed=train_fr + val_fr,
        seed=seed + 1,
        cache_dir=cache_dir,
    )

    train_pairs = [(text, 0) for text in en_texts[:train_en]] + [
        (text, 1) for text in fr_texts[:train_fr]
    ]
    val_pairs = [(text, 0) for text in en_texts[train_en : train_en + val_en]] + [
        (text, 1) for text in fr_texts[train_fr : train_fr + val_fr]
    ]

    # Shuffle so mini-batches contain mixed languages.
    random.Random(seed).shuffle(train_pairs)
    random.Random(seed + 1).shuffle(val_pairs)

    train_texts = [text for text, _ in train_pairs]
    train_labels = [label for _, label in train_pairs]
    val_texts = [text for text, _ in val_pairs]
    val_labels = [label for _, label in val_pairs]
    return train_texts, train_labels, val_texts, val_labels


def build_vocab(texts: list[str]) -> dict[str, int]:
    """Create a capped character vocabulary ordered by frequency."""
    counter = Counter("".join(texts))
    vocab = list(SPECIAL_TOKENS)

    remaining_slots = MAX_CHAR_VOCAB - len(vocab)
    common_chars = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[
        :remaining_slots
    ]
    vocab.extend(char for char, _ in common_chars)
    return {token: idx for idx, token in enumerate(vocab)}


def encode_texts(
    texts: list[str], labels: list[int], vocab: dict[str, int], max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw text to fixed-length character IDs plus class labels."""
    pad = vocab["<pad>"]
    bos = vocab["<bos>"]
    eos = vocab["<eos>"]
    unk = vocab["<unk>"]

    rows: list[list[int]] = []
    for text in texts:
        ids = [bos] + [vocab.get(ch, unk) for ch in text] + [eos]

        # Truncate long text but keep EOS at the end of the kept window.
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
            ids[-1] = eos
        else:
            ids.extend([pad] * (max_seq_len - len(ids)))
        rows.append(ids)

    return (
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def pooled_logits(model: StackedSLiCE, x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Run token IDs through the encoder and mean-pool non-pad positions."""
    hidden = model.embedding(x.long())
    for layer in model.layers:
        hidden = layer(hidden)

    mask = (x != pad_id).unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return model.linear(pooled)


def vf_A_penalty(model: StackedSLiCE, device: torch.device) -> torch.Tensor:
    """L2-regularise the A matrices in SLiCE to reduce unstable dynamics."""
    penalty = torch.zeros((), device=device)
    for layer in model.layers:
        sl = layer.slice
        if hasattr(sl, "vf_A"):
            penalty = penalty + sl.vf_A.weight.pow(2).mean()
        else:
            penalty = (
                penalty
                + sl.vf_A_diag.weight.pow(2).mean()
                + sl.vf_A_dense.weight.pow(2).mean()
            )
    return penalty


def train_step(
    model: StackedSLiCE,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    device: torch.device,
    optimiser: torch.optim.Optimizer,
    a_penalty: float,
    pad_id: int,
) -> tuple[float, float]:
    """Run one optimisation step on a sampled mini-batch."""
    model.train(True)
    xb = xb.to(device)
    yb = yb.to(device)

    logits = pooled_logits(model, xb, pad_id)
    cls_loss = F.cross_entropy(logits, yb)
    loss = cls_loss + a_penalty * vf_A_penalty(model, device)

    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimiser.step()

    acc = (logits.argmax(dim=-1) == yb).float().mean().item()
    return loss.item(), acc


@torch.inference_mode()
def evaluate(
    model: StackedSLiCE,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    pad_id: int,
) -> tuple[float, float]:
    """Compute full-set loss/accuracy with no gradient tracking."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(device)
        yb = y[start : start + batch_size].to(device)

        logits = pooled_logits(model, xb, pad_id)
        loss = F.cross_entropy(logits, yb)

        total_loss += loss.item() * yb.shape[0]
        total_correct += (logits.argmax(dim=-1) == yb).sum().item()
        total_items += yb.shape[0]

    return total_loss / total_items, total_correct / total_items


@torch.inference_mode()
def show_predictions(
    model: StackedSLiCE,
    x: torch.Tensor,
    y: torch.Tensor,
    texts: list[str],
    *,
    device: torch.device,
    pad_id: int,
    n: int = 6,
) -> None:
    """Print a few predictions to inspect confidence and errors."""
    model.eval()
    probs = pooled_logits(model, x[:n].to(device), pad_id).softmax(dim=-1).cpu()

    print("\nSample predictions:")
    for i in range(min(n, len(texts))):
        pred = int(probs[i].argmax())
        true = int(y[i])
        conf = float(probs[i, pred])
        snippet = texts[i][:100].replace("\n", " ")
        print(
            f"  text='{snippet}' | true={LANGUAGES[true]} "
            f"pred={LANGUAGES[pred]} p={conf:.3f}"
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.eval_every < 1:
        raise ValueError("--eval-every must be >= 1")
    if args.max_seq_len < 1:
        raise ValueError("--max-seq-len must be >= 1")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    # SLiCEs scale the input path; 1 / sequence length is a stable default.
    scale = 1.0 / float(args.max_seq_len)

    # 1) Load and label raw training/validation text.
    train_texts, train_labels, val_texts, val_labels = load_wikipedia_en_fr(
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )

    # 2) Build a training vocabulary and encode both splits.
    vocab = build_vocab(train_texts)
    pad_id = vocab["<pad>"]
    x_train, y_train = encode_texts(train_texts, train_labels, vocab, args.max_seq_len)
    x_val, y_val = encode_texts(val_texts, val_labels, vocab, args.max_seq_len)

    # 3) Create model and optimiser.
    model = StackedSLiCE(
        num_layers=NUM_LAYERS,
        data_dim=len(vocab),
        hidden_dim=HIDDEN_DIM,
        label_dim=2,
        tokens=True,
        block_size=BLOCK_SIZE,
        diagonal_dense=False,
        scale=scale,
        use_glu=True,
        dropout_rate=DROPOUT,
        use_parallel=True,
        chunk_size=128,
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("Task: character-level language disambiguation")
    print("Dataset: wikimedia/wikipedia (20231101.en + 20231101.fr)")
    print(f"Train: {x_train.shape[0]} | Val: {x_val.shape[0]}")
    print(f"Vocab size: {len(vocab)} | Seq len: {args.max_seq_len}")
    print(f"Device: {device}\n")

    # 4) Sample mini-batches and periodically evaluate.
    running_loss = 0.0
    running_acc = 0.0
    steps_since_eval = 0

    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, x_train.shape[0], (args.batch_size,))
        xb = x_train[idx]
        yb = y_train[idx]

        loss, acc = train_step(
            model,
            xb,
            yb,
            device=device,
            optimiser=optimiser,
            a_penalty=args.a_penalty,
            pad_id=pad_id,
        )
        running_loss += loss
        running_acc += acc
        steps_since_eval += 1

        if step % args.eval_every == 0 or step == args.train_steps:
            mean_train_loss = running_loss / steps_since_eval
            mean_train_acc = running_acc / steps_since_eval
            val_loss, val_acc = evaluate(
                model,
                x_val,
                y_val,
                batch_size=args.batch_size,
                device=device,
                pad_id=pad_id,
            )
            print(
                f"step {step:05d} | "
                f"train_loss={mean_train_loss:.4f} train_acc={mean_train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )
            running_loss = 0.0
            running_acc = 0.0
            steps_since_eval = 0

    # 5) Show a few validation predictions.
    show_predictions(
        model,
        x_val,
        y_val,
        val_texts,
        device=device,
        pad_id=pad_id,
    )


if __name__ == "__main__":
    main()
