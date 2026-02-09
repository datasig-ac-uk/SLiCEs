from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from slices import SLiCE

MODES = ("diagonal", "block_diagonal", "diagonal_dense", "dense")
MODE_TITLES = {
    "diagonal": "Diagonal",
    "block_diagonal": "Block-Diagonal",
    "diagonal_dense": "Diagonal + Dense",
    "dense": "Dense",
}

Z_BREAK_EVEN = 1.0
EPS = 1e-3
CLIP_TOL = 1e-9
Z_FLOOR = 1e-6
REFINE_FACTOR = 8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SLiCE recurrent vs parallel forward pass over a grid of "
            "sequence lengths and hidden dimensions for four SLiCE modes, "
            "then generate "
            "a combined 2x2 set of 3D speedup plots."
        )
    )
    arg_specs = [
        (
            "--seq-lens",
            dict(
                type=int,
                nargs="+",
                default=[64, 128, 256, 512, 1024],
                help="Sequence lengths to benchmark.",
            ),
        ),
        (
            "--dims",
            dict(
                type=int,
                nargs="+",
                default=[16, 32, 64, 128, 256],
                help="Input/hidden dimensions to benchmark.",
            ),
        ),
        ("--batch-size", dict(type=int, default=16, help="Batch size.")),
        ("--warmup", dict(type=int, default=3, help="Warmup iterations.")),
        ("--iters", dict(type=int, default=10, help="Timed iterations.")),
        (
            "--device",
            dict(
                type=str,
                default="cuda" if torch.cuda.is_available() else "cpu",
                choices=["cpu", "cuda"],
                help="Device to benchmark on.",
            ),
        ),
        (
            "--output-dir",
            dict(
                type=Path,
                default=Path("slices/examples/images"),
                help="Directory where the combined 3D speedup plot is saved.",
            ),
        ),
        ("--seed", dict(type=int, default=0, help="Random seed for reproducibility.")),
        (
            "--elev",
            dict(
                type=float, default=24.0, help="3D camera elevation angle in degrees."
            ),
        ),
        (
            "--azim",
            dict(type=float, default=-60.0, help="3D camera azimuth angle in degrees."),
        ),
    ]
    for name, kwargs in arg_specs:
        parser.add_argument(name, **kwargs)
    return parser.parse_args()


@torch.inference_mode()
def _benchmark_forward(
    model: SLiCE,
    x: torch.Tensor,
    *,
    parallel: bool,
    chunk_size: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    sync = (
        (lambda: torch.cuda.synchronize(device))
        if device.type == "cuda"
        else (lambda: None)
    )
    for _ in range(warmup):
        model(x, parallel=parallel, chunk_size=chunk_size)
    sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        model(x, parallel=parallel, chunk_size=chunk_size)
    sync()
    t1 = time.perf_counter()

    return (t1 - t0) / iters


def _to_log2(z, floor: float = 1e-12):
    return np.log2(np.clip(z, floor, None))


def _clip_triangle_to_halfspace(
    tri,
    *,
    z_plane: float,
    keep_below: bool,
    tol: float,
) -> list[np.ndarray]:
    # Sutherland-Hodgman clipping against the horizontal plane z=z_plane.
    def intersect(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        z0 = float(p0[2])
        dz = float(p1[2]) - z0
        if abs(dz) <= tol:
            return 0.5 * (p0 + p1)
        t = (z_plane - z0) / dz
        t = max(0.0, min(1.0, t))
        return p0 + t * (p1 - p0)

    def inside(p: np.ndarray) -> bool:
        if keep_below:
            return float(p[2]) <= z_plane + tol
        return float(p[2]) >= z_plane - tol

    poly: list[np.ndarray] = tri
    out: list[np.ndarray] = []
    for i, start in enumerate(poly):
        end = poly[(i + 1) % len(poly)]
        s_in = inside(start)
        e_in = inside(end)

        if s_in and e_in:
            out.append(end)
        elif s_in and not e_in:
            out.append(intersect(start, end))
        elif (not s_in) and e_in:
            out.append(intersect(start, end))
            out.append(end)

    return out


def _split_surface_by_plane(
    x_grid,
    y_grid,
    z_grid,
    *,
    z_plane: float,
    tol: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    # Split each grid cell into two triangles, then clip triangles into
    # below/above pieces.
    def fan_triangulate(poly):
        if len(poly) < 3:
            return []
        anchor = poly[0]
        return [
            np.vstack((anchor, poly[i], poly[i + 1])) for i in range(1, len(poly) - 1)
        ]

    below_tris = []
    above_tris = []

    rows, cols = z_grid.shape
    for r in range(rows - 1):
        for c in range(cols - 1):
            p00 = np.array([x_grid[r, c], y_grid[r, c], z_grid[r, c]], dtype=np.float64)
            p10 = np.array(
                [x_grid[r + 1, c], y_grid[r + 1, c], z_grid[r + 1, c]], dtype=np.float64
            )
            p11 = np.array(
                [x_grid[r + 1, c + 1], y_grid[r + 1, c + 1], z_grid[r + 1, c + 1]],
                dtype=np.float64,
            )
            p01 = np.array(
                [x_grid[r, c + 1], y_grid[r, c + 1], z_grid[r, c + 1]], dtype=np.float64
            )

            cell_tris = ([p00, p10, p11], [p00, p11, p01])
            for tri in cell_tris:
                if not all(np.isfinite(v[2]) for v in tri):
                    continue
                below_poly = _clip_triangle_to_halfspace(
                    tri,
                    z_plane=z_plane,
                    keep_below=True,
                    tol=tol,
                )
                above_poly = _clip_triangle_to_halfspace(
                    tri,
                    z_plane=z_plane,
                    keep_below=False,
                    tol=tol,
                )
                below_tris.extend(fan_triangulate(below_poly))
                above_tris.extend(fan_triangulate(above_poly))

    return below_tris, above_tris


def _refine_surface_grid(
    x_grid,
    y_grid,
    z_grid,
    *,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if factor <= 1:
        return x_grid, y_grid, z_grid

    x_old = x_grid[0, :]
    y_old = y_grid[:, 0]
    x_new = np.linspace(
        float(x_old[0]), float(x_old[-1]), (len(x_old) - 1) * factor + 1
    )
    y_new = np.linspace(
        float(y_old[0]), float(y_old[-1]), (len(y_old) - 1) * factor + 1
    )

    z_x = np.vstack([np.interp(x_new, x_old, row) for row in z_grid])
    z_new = np.vstack(
        [np.interp(y_new, y_old, z_x[:, j]) for j in range(z_x.shape[1])]
    ).T
    x_refined, y_refined = np.meshgrid(x_new, y_new)
    return x_refined, y_refined, z_new


def _build_combined_plot(
    seq_lens,
    dims,
    mode_speedups,
    outpath: Path,
    elev: float,
    azim: float,
    zmin: float,
    zmax: float,
) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import cm, colors
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it and rerun, "
            "e.g. `uv add --dev matplotlib`."
        ) from exc

    # mplot3d has poor 3D log rendering; transform coordinates and relabel ticks.
    x_vals = np.log10(np.asarray(seq_lens, dtype=np.float64))
    y_vals = np.log10(np.asarray(dims, dtype=np.float64))
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    zmin = min(zmin, Z_BREAK_EVEN)
    zmax = max(zmax, Z_BREAK_EVEN)
    norm = colors.Normalize(vmin=zmin, vmax=zmax)
    zmin_plot = float(_to_log2(zmin))
    zmax_plot = float(_to_log2(zmax))
    fig = plt.figure(figsize=(16, 12))
    viridis = matplotlib.colormaps["viridis"]

    def _add_tri_collection(
        ax,
        triangles,
        *,
        edgecolor: str = "none",
        linewidth: float = 0.0,
        alpha: float = 1.0,
        solid_color: Optional[str] = None,
    ) -> None:
        if not triangles:
            return
        if solid_color is None:
            face_vals = np.array(
                [float(np.mean(2.0 ** tri[:, 2])) for tri in triangles],
                dtype=np.float64,
            )
            facecolors = viridis(norm(face_vals))
            facecolors[:, 3] = alpha
        else:
            face_rgba = np.array(colors.to_rgba(solid_color), dtype=np.float64)
            face_rgba[3] = alpha
            facecolors = np.repeat(face_rgba[np.newaxis, :], len(triangles), axis=0)

        collection = Poly3DCollection(
            triangles,
            facecolors=facecolors,
            edgecolors=edgecolor,
            linewidths=linewidth,
            antialiased=True,
        )
        ax.add_collection3d(collection)

    def _style_axis(ax) -> None:
        ax.set_xlabel("Sequence Length", fontsize=16, labelpad=14)
        ax.set_ylabel("Hidden Dimension", fontsize=16, labelpad=10)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel("Speedup", fontsize=16, rotation=90, labelpad=16)
        ax.set_xlim(x_vals.min(), x_vals.max())
        ax.set_ylim(y_vals.max(), y_vals.min())
        ax.set_zlim(zmin_plot, zmax_plot)
        ax.set_zticks([-1, 0, 1, 2, 3, 4], ["0.5", "1", "2", "4", "8", "16"])
        ax.set_xticks(x_vals, [str(v) for v in seq_lens])
        ax.set_yticks(y_vals, [str(v) for v in dims])
        ax.tick_params(axis="x", which="major", labelsize=14, pad=1)
        ax.tick_params(axis="y", which="major", labelsize=14, pad=1)
        ax.tick_params(axis="z", which="major", labelsize=14)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment("left")
        for lbl in ax.get_yticklabels():
            lbl.set_horizontalalignment("right")
        ax.set_proj_type("persp")
        ax.set_box_aspect((1.35, 1.1, 0.9))
        ax.view_init(elev=elev, azim=azim - 90.0)

    for idx, mode in enumerate(MODES, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        z_grid = np.where(np.isfinite(mode_speedups[mode]), mode_speedups[mode], np.nan)
        z_grid = np.clip(z_grid, Z_FLOOR, None)
        x_refined, y_refined, z_refined = _refine_surface_grid(
            x_grid,
            y_grid,
            z_grid,
            factor=REFINE_FACTOR,
        )

        speed_below, speed_above = _split_surface_by_plane(
            x_refined,
            y_refined,
            z_refined,
            z_plane=Z_BREAK_EVEN,
            tol=CLIP_TOL,
        )
        red_top = [
            np.column_stack(
                (tri[:, 0], tri[:, 1], np.full(3, Z_BREAK_EVEN + EPS, dtype=np.float64))
            )
            for tri in speed_below
        ]
        red_bottom = [
            np.column_stack(
                (tri[:, 0], tri[:, 1], np.full(3, Z_BREAK_EVEN - EPS, dtype=np.float64))
            )
            for tri in speed_above
        ]

        def log2_tris(tris):
            return [np.column_stack((tri[:, :2], _to_log2(tri[:, 2]))) for tri in tris]

        speed_below_plot = log2_tris(speed_below)
        speed_above_plot = log2_tris(speed_above)
        red_top_plot = log2_tris(red_top)
        red_bottom_plot = log2_tris(red_bottom)

        # Plot four parts independently: red-below, speed-below, speed-above, red-above.
        _add_tri_collection(ax, red_bottom_plot, alpha=0.35, solid_color="red")
        _add_tri_collection(
            ax, speed_below_plot, edgecolor="none", linewidth=0.0, alpha=0.98
        )
        _add_tri_collection(
            ax, speed_above_plot, edgecolor="none", linewidth=0.0, alpha=0.98
        )
        _add_tri_collection(ax, red_top_plot, alpha=0.35, solid_color="red")

        # Intersection curve for speedup == 1.
        ax.contour(
            x_refined,
            y_refined,
            np.ma.masked_invalid(_to_log2(z_refined)),
            levels=[float(_to_log2(Z_BREAK_EVEN))],
            colors=["red"],
            linewidths=2.0,
            linestyles=["-"],
            offset=None,
        )

        ax.scatter(
            x_grid.flatten(),
            y_grid.flatten(),
            _to_log2(z_grid.flatten()),
            c="k",
            s=10,
            alpha=0.75,
        )

        ax.set_title(MODE_TITLES[mode], fontsize=19, pad=3, y=0.975)
        _style_axis(ax)

    fig.suptitle("SLiCE Parallel Speedup Over Recurrent", fontsize=22, y=0.985)
    sm = cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=fig.axes, shrink=0.68, aspect=30, pad=0.20, label="Speedup"
    )
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Speedup", fontsize=16)
    cbar.ax.set_position([0.90, 0.18, 0.018, 0.64])
    fig.text(
        0.855,
        0.84,
        "Red plane: break-even\nboundary (speedup = 1)",
        fontsize=13,
        color="red",
        ha="left",
        va="center",
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(
        left=0.03, right=0.82, bottom=0.04, top=0.93, wspace=0.14, hspace=0.18
    )
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _pick_block_size(mode: str, hidden_dim: int):
    if mode == "diagonal":
        return 1
    if mode == "dense":
        return hidden_dim
    if mode in ("block_diagonal", "diagonal_dense"):
        return 4 if hidden_dim > 4 and hidden_dim % 4 == 0 else None
    raise ValueError(f"Unknown mode: {mode}")


def _make_model(mode: str, dim: int, device: torch.device):
    block_size = _pick_block_size(mode, dim)
    if block_size is None:
        return None

    model = SLiCE(
        input_dim=dim,
        hidden_dim=dim,
        block_size=block_size,
        diagonal_dense=(mode == "diagonal_dense"),
        bias=True,
    ).to(device)
    model.eval()
    return model


def _benchmark_mode_grid(
    mode: str,
    *,
    dims,
    seq_lens,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    speedups = np.full((len(dims), len(seq_lens)), np.nan, dtype=np.float64)
    for d_idx, dim in enumerate(dims):
        model = _make_model(mode, dim, device)
        if model is None:
            print(f"dim={dim:4d} | skipped (no valid block_size for mode={mode})")
            continue

        for s_idx, seq_len in enumerate(seq_lens):
            x = torch.randn(
                args.batch_size, seq_len, dim, device=device, dtype=torch.float32
            )
            recurrent_t = _benchmark_forward(
                model,
                x,
                parallel=False,
                chunk_size=seq_len,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            parallel_t = _benchmark_forward(
                model,
                x,
                parallel=True,
                chunk_size=seq_len,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )

            speedup = recurrent_t / parallel_t if parallel_t > 0 else math.inf
            speedups[d_idx, s_idx] = speedup
            print(
                f"dim={dim:4d}, seq_len={seq_len:5d} | "
                f"recurrent={recurrent_t:.6f}s, parallel={parallel_t:.6f}s, "
                f"speedup={speedup:.3f}x"
            )
    return speedups


def _speedup_bounds(mode_speedups):
    finite = [
        arr[np.isfinite(arr)]
        for arr in mode_speedups.values()
        if np.isfinite(arr).any()
    ]
    finite_all = np.concatenate(finite)
    return max(Z_FLOOR, float(finite_all.min())), float(finite_all.max())


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be >= 0 and iters must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    seq_lens = sorted(set(args.seq_lens))
    dims = sorted(set(args.dims))

    print("Benchmarking recurrent vs parallel...")
    print(
        f"device={device.type}, batch_size={args.batch_size}, chunk_size=seq_len, "
        f"warmup={args.warmup}, iters={args.iters}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    mode_speedups: dict[str, np.ndarray] = {}

    for mode in MODES:
        print(f"\nMode: {mode}")
        mode_speedups[mode] = _benchmark_mode_grid(
            mode,
            dims=dims,
            seq_lens=seq_lens,
            args=args,
            device=device,
        )

    zmin, zmax = _speedup_bounds(mode_speedups)

    outpath = args.output_dir / "parallel_vs_recurrent_speedup_3d_all_modes.png"
    _build_combined_plot(
        seq_lens,
        dims,
        mode_speedups,
        outpath,
        elev=args.elev,
        azim=args.azim,
        zmin=zmin,
        zmax=zmax,
    )
    print(f"Saved combined 3D speedup plot to: {outpath}")


if __name__ == "__main__":
    main()
