"""
Train a self-supervised stacked SLiCE language model, export hidden states,
and optionally plot UMAP projections from the exported embeddings.

Outputs:
  <export-path>.dat  : numpy memmap of hidden states
  <export-path>.npz  : metadata (shape, dtype, labels)
"""

from __future__ import annotations

import argparse
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from slices import StackedSLiCE

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
MAX_CHAR_VOCAB = 256
MULTI_HORIZONS = (1, 2, 4)
LANGUAGE_LABELS = (
    (0, "English", "tab:blue"),
    (1, "French", "tab:orange"),
)
DEFAULT_TIME_INDICES = "1,20,100,-5,-1,0"
EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "outputs" / "language_selfsupervised"
DEFAULT_EXPORT_PATH = DEFAULT_OUTPUT_DIR / "slice_embeddings"
DEFAULT_PLOT_PATH = DEFAULT_OUTPUT_DIR / "slice_hidden_umap.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a self-supervised SLiCE language model on streaming "
            "Wikipedia text, export hidden states, and optionally plot UMAPs."
        )
    )

    parser.add_argument("--n-per-lang", type=int, default=8000)
    parser.add_argument("--max-seq-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )

    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=192)

    parser.add_argument("--horizon", choices=["single", "multi"], default="single")
    parser.add_argument("--export-mode", choices=["all", "final"], default="all")
    parser.add_argument("--export-path", type=str, default=str(DEFAULT_EXPORT_PATH))

    parser.add_argument("--plot-umap", action="store_true")
    parser.add_argument(
        "--umap-backend",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Use CPU UMAP by default, or CUDA UMAP if RAPIDS is installed.",
    )
    parser.add_argument(
        "--plot-time-indices",
        type=str,
        default=DEFAULT_TIME_INDICES,
        help=(
            "Comma-separated indices: 0 means the first pad step after the last "
            "valid hidden state, negative values are relative to that point, and "
            "positive values are one-indexed absolute timesteps."
        ),
    )
    parser.add_argument("--plot-max-points", type=int, default=4000)
    parser.add_argument("--plot-path", type=str, default=str(DEFAULT_PLOT_PATH))
    parser.add_argument("--umap-neighbors", type=int, default=80)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def parse_time_indices(raw: str) -> list[int]:
    indices = [item.strip() for item in raw.split(",") if item.strip()]
    if not indices:
        raise ValueError("plot-time-indices must contain at least one integer.")
    return [int(item) for item in indices]


def final_state_indices(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Index of the last hidden state used for next-token prediction."""
    targets = x[:, 1:]
    return targets.ne(pad_id).sum(dim=1).clamp_min(1) - 1


def resolve_time_positions(
    time_index: int,
    final_indices: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a plot index to per-sequence hidden-state positions."""
    if time_index <= 0:
        positions = final_indices + time_index + 1
        valid = (positions >= 0) & (positions < seq_len)
    else:
        positions = np.full_like(final_indices, time_index - 1)
        valid = positions <= final_indices
    return positions, valid


def collect_wikipedia_texts(
    config_name: str,
    *,
    needed: int,
    seed: int,
    cache_dir: str | None,
) -> list[str]:
    from datasets import load_dataset

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
    n_per_lang: int,
    *,
    seed: int,
    cache_dir: str | None,
) -> tuple[list[str], torch.Tensor]:
    en_texts = collect_wikipedia_texts(
        "20231101.en",
        needed=n_per_lang,
        seed=seed,
        cache_dir=cache_dir,
    )
    fr_texts = collect_wikipedia_texts(
        "20231101.fr",
        needed=n_per_lang,
        seed=seed + 1,
        cache_dir=cache_dir,
    )

    pairs = [(text, 0) for text in en_texts] + [(text, 1) for text in fr_texts]
    random.Random(seed).shuffle(pairs)

    texts = [text for text, _ in pairs]
    labels = torch.tensor([label for _, label in pairs], dtype=torch.long)
    return texts, labels


def build_vocab(texts: list[str]) -> dict[str, int]:
    counter = Counter("".join(texts))
    vocab = list(SPECIAL_TOKENS)
    remaining_slots = MAX_CHAR_VOCAB - len(vocab)

    common_chars = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[
        :remaining_slots
    ]
    vocab.extend(char for char, _ in common_chars)
    return {token: idx for idx, token in enumerate(vocab)}


def encode_texts(texts: list[str], vocab: dict[str, int], max_len: int) -> torch.Tensor:
    pad = vocab["<pad>"]
    bos = vocab["<bos>"]
    eos = vocab["<eos>"]
    unk = vocab["<unk>"]

    rows: list[list[int]] = []
    for text in texts:
        ids = [bos] + [vocab.get(char, unk) for char in text] + [eos]
        if len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = eos
        else:
            ids.extend([pad] * (max_len - len(ids)))
        rows.append(ids)

    return torch.tensor(rows, dtype=torch.long)


class SelfSupervisedSLiCE(nn.Module):
    """Thin wrapper around StackedSLiCE providing a horizon-aware loss."""

    def __init__(self, vocab_size: int, args: argparse.Namespace, pad_id: int):
        super().__init__()
        self.pad_id = pad_id

        if args.horizon == "single":
            label_dim = vocab_size
        else:
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
        return self.model(x)

    def hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.hidden(x)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(x)
        targets = x[:, 1:]

        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
            horizons = (1,)
        else:
            horizons = MULTI_HORIZONS

        total = 0.0
        for out, horizon in zip(outputs, horizons, strict=True):
            readout = out[:, :-horizon]
            target = targets[:, horizon - 1 :]
            mask = target.ne(self.pad_id)

            loss = F.cross_entropy(
                readout.reshape(-1, readout.size(-1)),
                target.reshape(-1),
                reduction="none",
            ).view_as(target)

            total += (loss * mask).sum() / mask.sum().clamp_min(1)

        return total / len(outputs)


def extract_final(states: torch.Tensor, x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Extract the final hidden state used for next-token prediction."""
    idx = final_state_indices(x, pad_id)
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
    """Memory-safe export of hidden states using a numpy memmap."""
    assert export_mode in {"final", "all"}
    assert dtype in {"float32", "float16"}

    total_examples, seq_len = x_all.shape
    shape = (
        (total_examples, hidden_dim)
        if export_mode == "final"
        else (total_examples, seq_len, hidden_dim)
    )
    final_indices = final_state_indices(x_all, pad_id).cpu().numpy()

    mmap_path = export_path + ".dat"
    meta_path = export_path + ".npz"
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    print(f"Creating memmap {mmap_path} with shape {shape}")
    mmap = np.memmap(mmap_path, dtype=dtype, mode="w+", shape=shape)

    write_idx = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, total_examples, batch_size):
            xb = x_all[start : start + batch_size].to(device)
            hidden = model.hidden_states(xb)

            out = (
                extract_final(hidden, xb, pad_id) if export_mode == "final" else hidden
            )
            out_np = out.to(dtype=torch_dtype).cpu().numpy()
            batch = out_np.shape[0]

            mmap[write_idx : write_idx + batch] = out_np
            write_idx += batch

            if device.type == "cuda":
                torch.cuda.empty_cache()
            if write_idx % (batch_size * 10) == 0:
                print(f"  wrote {write_idx}/{total_examples}")

    mmap.flush()
    del mmap

    np.savez(
        meta_path,
        shape=shape,
        dtype=dtype,
        labels=labels.numpy(),
        export_mode=export_mode,
        hidden_dim=hidden_dim,
        final_indices=final_indices,
    )

    print("Export complete:")
    print(f"  data: {mmap_path}")
    print(f"  meta: {meta_path}")


def project_umap(
    embeddings: np.ndarray,
    *,
    backend: str,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> tuple[np.ndarray, str]:
    umap_kwargs = dict(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )

    if backend in {"auto", "cuda"}:
        try:
            import cupy as cp
            from cuml.manifold import UMAP as CudaUMAP
        except ImportError:
            if backend == "cuda":
                raise RuntimeError(
                    "CUDA UMAP requires `cupy` and `cuml`. Install the "
                    "optional CUDA example dependencies with "
                    "`uv sync --group examples --group cuda-umap`, then rerun "
                    "with `--umap-backend cuda`."
                ) from None
        else:
            projection = CudaUMAP(**umap_kwargs).fit_transform(cp.asarray(embeddings))
            return cp.asnumpy(projection), "cuda"

    try:
        from umap import UMAP as CpuUMAP
    except ImportError as exc:
        raise RuntimeError(
            "UMAP plotting requires `umap-learn`. Install example dependencies "
            "with `uv sync --group examples`."
        ) from exc

    projection = CpuUMAP(**umap_kwargs, n_jobs=1).fit_transform(embeddings)
    return projection, "cpu"


def plot_umap_from_export(
    *,
    export_path: str,
    plot_path: str,
    time_indices: list[int],
    backend: str,
    max_points: int,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires `matplotlib`. Install example dependencies with "
            "`uv sync --group examples`."
        ) from exc

    meta_path = export_path + ".npz"
    data_path = export_path + ".dat"
    meta = np.load(meta_path, allow_pickle=True)
    shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["dtype"].item())
    labels = np.asarray(meta["labels"])
    export_mode = str(meta["export_mode"].item())
    final_indices = (
        np.asarray(meta["final_indices"], dtype=np.int64)
        if "final_indices" in meta.files
        else None
    )

    embeddings = np.memmap(data_path, dtype=dtype, mode="r", shape=shape)
    if max_points > 0 and len(labels) > max_points:
        rng = np.random.default_rng(seed)
        sample_idx = np.sort(rng.choice(len(labels), size=max_points, replace=False))
    else:
        sample_idx = np.arange(len(labels))

    sampled_labels = labels[sample_idx]
    if embeddings.ndim == 2:
        panels = [
            (
                r"SLiCE $H_{T-1}$",
                np.asarray(embeddings[sample_idx]),
                sampled_labels,
            )
        ]
    else:
        if final_indices is None:
            raise RuntimeError(
                "This export is missing per-sequence final-state metadata. "
                "Re-run the script to regenerate the embeddings before plotting."
            )
        panels = []
        sampled_final_indices = final_indices[sample_idx]
        for time_index in time_indices:
            positions, valid = resolve_time_positions(
                time_index,
                sampled_final_indices,
                embeddings.shape[1],
            )
            if not np.any(valid):
                raise IndexError(
                    f"time index {time_index} is out of range for all sampled sequences"
                )
            panels.append(
                (
                    (
                        r"SLiCE $H_{T}$"
                        if time_index == 0
                        else (
                            rf"SLiCE $H_{{T{time_index}}}$"
                            if time_index < 0
                            else rf"SLiCE $H_{{t={time_index}}}$"
                        )
                    ),
                    np.asarray(embeddings[sample_idx[valid], positions[valid], :]),
                    sampled_labels[valid],
                )
            )

    n_panels = len(panels)
    ncols = min(3, n_panels)
    nrows = math.ceil(n_panels / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axs_flat = np.atleast_1d(axs).ravel()

    backend_used = None
    for ax, (title, states, panel_labels) in zip(axs_flat, panels, strict=True):
        projection, backend_used = project_umap(
            states,
            backend=backend,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            seed=seed,
        )

        for label, name, colour in LANGUAGE_LABELS:
            mask = panel_labels == label
            ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                s=4,
                alpha=0.4,
                c=colour,
                label=name,
                linewidths=0,
            )

        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(markerscale=2)

    for ax in axs_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        f"Temporal evolution of SLiCE hidden states ({backend_used.upper()} UMAP)"
        if export_mode == "all"
        else f"SLiCE hidden state projection ({backend_used.upper()} UMAP)",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(plot_path, dpi=200)
    print(f"Saved figure to {plot_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    cache_dir = None if args.cache_dir is None else str(Path(args.cache_dir))
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    texts, labels = load_wikipedia_en_fr(
        args.n_per_lang,
        seed=args.seed,
        cache_dir=cache_dir,
    )
    vocab = build_vocab(texts)
    x_all = encode_texts(texts, vocab, args.max_seq_len)

    print(f"Loaded {x_all.size(0)} sequences with vocabulary size {len(vocab)}")

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

    if args.plot_umap:
        plot_umap_from_export(
            export_path=args.export_path,
            plot_path=args.plot_path,
            time_indices=parse_time_indices(args.plot_time_indices),
            backend=args.umap_backend,
            max_points=args.plot_max_points,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
