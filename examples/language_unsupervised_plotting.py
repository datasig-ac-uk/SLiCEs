"""
UMAP visualisation of SLiCE hidden states across time.

Assumes:
  - embeddings stored as a numpy memmap: (N, T, D)
  - labels are binary (e.g. English / French)
  - time index is relative to the *end* of the sequence
"""

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from cuml.manifold import UMAP

# ---------------------------------------------------------------------
# Load embeddings + metadata
# ---------------------------------------------------------------------

META_PATH = "slice_embeddings.npz"
DATA_PATH = "slice_embeddings.dat"

meta = np.load(META_PATH, allow_pickle=True)
shape = tuple(meta["shape"])
dtype = meta["dtype"].item()
labels = meta["labels"]

embeddings = np.memmap(
    DATA_PATH,
    dtype=dtype,
    mode="r",
    shape=shape,
)

assert embeddings.ndim == 3, "Expected embeddings of shape (N, T, D)"
N, T, D = embeddings.shape

print(f"Loaded embeddings: N={N}, T={T}, D={D}, dtype={dtype}")


# ---------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------

# Time indices relative to the end of the sequence
# -1 = H_{T-1}, -2 = H_{T-2}, etc.
TIME_INDICES = [0,20,100,-5,-2,-1]

LABELS = [
    (0, "English", "tab:blue"),
    (1, "French", "tab:orange"),
]

UMAP_KWARGS = dict(
    n_components=2,
    n_neighbors=80,
    min_dist=0.1,
    metric="cosine",
    random_state=0,
)

FIGSIZE = (15, 10)
POINT_SIZE = 4
ALPHA = 0.4


# ---------------------------------------------------------------------
# Create figure
# ---------------------------------------------------------------------

fig, axs = plt.subplots(2, 3, figsize=FIGSIZE)
axs = axs.flatten()


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

for ax, t in zip(axs, TIME_INDICES, strict=True):
    if not (-T <= t < T):
        raise IndexError(f"time index {t} out of range for T={T}")

    # (N, D) slice at time t
    H = embeddings[:, t, :]

    # ---------------------------
    # Move to GPU
    # ---------------------------
    H_gpu = cp.asarray(H)

    # ---------------------------
    # UMAP (GPU)
    # ---------------------------
    umap = UMAP(**UMAP_KWARGS)
    Z_gpu = umap.fit_transform(H_gpu)

    # Back to CPU for plotting
    Z = cp.asnumpy(Z_gpu)

    # ---------------------------
    # Plot by label
    # ---------------------------
    for lab, name, colour in LABELS:
        m = labels == lab
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            s=POINT_SIZE,
            alpha=ALPHA,
            c=colour,
            label=name,
            linewidths=0,
        )

    # ---------------------------
    # Axis cosmetics
    # ---------------------------
    ax.set_title(rf"cuML UMAP of SLiCE $H_{{t={t}}}$")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=2)


# ---------------------------------------------------------------------
# Finalise and save
# ---------------------------------------------------------------------

fig.suptitle("Temporal evolution of SLiCE hidden states", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

OUT_PATH = "slice_hidden_umap.png"
fig.savefig(OUT_PATH, dpi=200)

print(f"Saved figure to {OUT_PATH}")
plt.show()
