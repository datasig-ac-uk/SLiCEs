import warnings
from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """A minimal RMSNorm used by the default SLiCELayer configuration."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms) * self.weight


class SLiCE(nn.Module):
    """
    A structured linear controlled differential equation (SLiCE) recurrence.

    Given a sequence of values (or increments) X_i in R^D for i=1,...,T, a SLiCE
    recurrence computes a sequence of hidden states y_i in R^H for i=1,...,T via the
    recurrence:
        y_i = y_{i-1} + A(X_i) y_{i-1} + B(X_i)   for i=1,...,T,
    where A: R^D -> R^{H x H} and B: R^D -> R^H are learnt linear functions and y_{0}
    is learnt.

    Args:
        input_dim (int): Dimensionality of the input features at each time step.
        hidden_dim (optional[int]): Dimensionality of the hidden state. If None, set to
                                    input_dim.
        bias (bool): If True, include the bias term B(X_i) in the recurrence.
        block_size (int): The size of the blocks along the diagonal of A.
        diagonal_dense (bool): If True, A is composed of a diagonal matrix and a single
                               dense block of size block_size x block_size.
        init_std (float): Standard deviation for vector field initialisation.
        scale (float): Scaling factor applied to the input.
        path_mode (str): Whether the input is treated as path values
                         ("values", default) or as increments
                         ("increments").

    Shape:
        - Input: (batch_size, seq_len, input_dim)
        - Output: (batch_size, seq_len, hidden_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        bias: bool = True,
        block_size: int = 4,
        diagonal_dense: bool = False,
        init_std: float = 0.01,
        scale: float = 1.0,
        input_dependent_init: bool = False,
        use_parallel: bool = True,
        chunk_size: int = 256,
        path_mode: str = "values",
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim
        if path_mode not in {"values", "increments"}:
            raise ValueError("path_mode must be one of {'values', 'increments'}.")
        if block_size < 1:
            raise ValueError("block_size must be at least 1.")
        if not diagonal_dense and hidden_dim % block_size != 0:
            raise ValueError("hidden_dim must be divisible by block_size.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.block_size = block_size
        self.init_std = init_std
        self.scale = scale
        self.input_dependent_init = input_dependent_init
        self.path_mode = path_mode

        self.use_parallel = use_parallel
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")
        self.chunk_size = int(chunk_size)
        if self.use_parallel:
            assoc_ops = getattr(torch, "_higher_order_ops", None)
            if assoc_ops is None or not hasattr(assoc_ops, "associative_scan"):
                warnings.warn(
                    "use_parallel=True requested, "
                    "but torch.associative_scan is unavailable. "
                    "Falling back to recurrent mode.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.use_parallel = False
            elif self.block_size >= 64 and hidden_dim >= 128:
                warnings.warn(
                    "Parallel mode may be slower than recurrent mode for large "
                    f"block_size ({self.block_size}) and hidden_dim ({hidden_dim}). "
                    "Consider setting use_parallel=False "
                    "if throughput regresses.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if diagonal_dense and self.block_size == self.hidden_dim:
            self.diagonal_dense = (
                False  # No point in diagonal + dense if only one block
            )
        elif diagonal_dense and self.block_size == 1:
            self.diagonal_dense = False  # No point in diagonal + dense if no dense part
        else:
            self.diagonal_dense = diagonal_dense

        # Define learnt initial hidden state y_0
        if self.input_dependent_init:
            self.init = nn.Linear(self.input_dim, self.hidden_dim)
            nn.init.normal_(self.init.weight, mean=0.0, std=self.init_std)
            if self.init.bias is not None:
                nn.init.zeros_(self.init.bias)
        else:
            self.init = torch.nn.Parameter(torch.randn(self.hidden_dim) * self.init_std)

        if self.diagonal_dense:
            # For diagonal + dense block structure, define separate parameters
            # for the diagonal and dense parts.
            self.vf_A_diag = nn.Linear(
                self.input_dim + 1, self.hidden_dim - self.block_size, bias=False
            )
            self.vf_A_dense = nn.Linear(
                self.input_dim + 1, self.block_size * self.block_size, bias=False
            )
            nn.init.normal_(self.vf_A_diag.weight, mean=0.0, std=self.init_std)
            nn.init.normal_(
                self.vf_A_dense.weight,
                mean=0.0,
                std=self.init_std / (self.block_size**0.5),
            )
        else:
            # Define the vector field A as a linear layer
            self.vf_A = nn.Linear(
                self.input_dim + 1, self.hidden_dim * self.block_size, bias=False
            )
            nn.init.normal_(
                self.vf_A.weight,
                mean=0.0,
                std=self.init_std / (self.block_size**0.5),
            )

        if bias:
            self.vf_B = nn.Linear(self.input_dim + 1, self.hidden_dim, bias=False)
            nn.init.normal_(self.vf_B.weight, mean=0.0, std=self.init_std)

    def _prepare_driving_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.path_mode == "values":
            return torch.diff(
                x,
                dim=1,
                prepend=torch.zeros_like(x[:, :1, :]),
            )
        return x

    def _prepare_augmented_inputs(self, x: torch.Tensor) -> torch.Tensor:
        path = self._prepare_driving_path(x)
        inc_ts = torch.ones(
            path.shape[0], path.shape[1], 1, device=x.device, dtype=x.dtype
        )
        return torch.cat((inc_ts, path), dim=-1) * self.scale

    # ---- scan kernels: block_size == 1 (elementwise) ----

    def _scan_kernels_elementwise(self) -> tuple[Callable, Callable, Callable]:
        def build(inp_chunk: torch.Tensor):
            A = self.vf_A(inp_chunk)  # (B, C, H)
            M = 1.0 + A
            if self.bias:
                b = self.vf_B(inp_chunk)
            else:
                b = torch.zeros_like(M)
            return (M, b)

        def combine(lhs, rhs):
            # Composition: rhs ∘ lhs
            M_l, b_l = lhs
            M_r, b_r = rhs
            M = M_r * M_l
            b = M_r * b_l + b_r
            return (M, b)

        def apply(prefix, y0: torch.Tensor):
            M, b = prefix  # (B, C, H)
            return M * y0.unsqueeze(1) + b

        return combine, build, apply

    # ---- scan kernels: block diagonal (block_size > 1, not diagonal_dense) ----

    def _scan_kernels_blockdiag(self) -> tuple[Callable, Callable, Callable]:
        bsz = self.block_size
        nblocks = self.hidden_dim // bsz

        def build(inp_chunk: torch.Tensor):
            # A: (B, C, nblocks, b, b)
            A = self.vf_A(inp_chunk).view(
                inp_chunk.shape[0], inp_chunk.shape[1], nblocks, bsz, bsz
            )
            eye = torch.eye(bsz, device=inp_chunk.device, dtype=inp_chunk.dtype).view(
                1, 1, 1, bsz, bsz
            )
            M = eye + A
            if self.bias:
                b = self.vf_B(inp_chunk).view(
                    inp_chunk.shape[0], inp_chunk.shape[1], nblocks, bsz
                )
            else:
                b = torch.zeros(
                    inp_chunk.shape[0],
                    inp_chunk.shape[1],
                    nblocks,
                    bsz,
                    device=inp_chunk.device,
                    dtype=inp_chunk.dtype,
                )
            return (M, b)

        def combine(lhs, rhs):
            M_l, b_l = lhs
            M_r, b_r = rhs
            M = torch.matmul(M_r, M_l)
            b = torch.matmul(M_r, b_l.unsqueeze(-1)).squeeze(-1) + b_r
            return (M, b)

        def apply(prefix, y0: torch.Tensor):
            M, b = prefix  # M: (B,C,nblocks,b,b), b: (B,C,nblocks,b)
            y0b = (
                y0.view(y0.shape[0], nblocks, bsz).unsqueeze(1).unsqueeze(-1)
            )  # (B,1,nblocks,b,1)
            y = torch.matmul(M, y0b).squeeze(-1) + b  # (B,C,nblocks,b)
            return y.reshape(y.shape[0], y.shape[1], self.hidden_dim)

        return combine, build, apply

    # ---- scan kernels: diagonal + one dense block ----

    def _scan_kernels_diagonal_dense(self) -> tuple[Callable, Callable, Callable]:
        bsz = self.block_size
        h = self.hidden_dim
        hdiag = h - bsz

        def build(inp_chunk: torch.Tensor):
            A_diag = self.vf_A_diag(inp_chunk)  # (B,C,hdiag)
            M_diag = 1.0 + A_diag

            A_dense = self.vf_A_dense(inp_chunk).view(
                inp_chunk.shape[0], inp_chunk.shape[1], bsz, bsz
            )
            eye = torch.eye(bsz, device=inp_chunk.device, dtype=inp_chunk.dtype).view(
                1, 1, bsz, bsz
            )
            M_dense = eye + A_dense

            if self.bias:
                B = self.vf_B(inp_chunk)  # (B,C,h)
                b_diag = B[..., :hdiag]
                b_dense = B[..., hdiag:]
            else:
                b_diag = torch.zeros_like(M_diag)
                b_dense = torch.zeros(
                    inp_chunk.shape[0],
                    inp_chunk.shape[1],
                    bsz,
                    device=inp_chunk.device,
                    dtype=inp_chunk.dtype,
                )

            return (M_diag, M_dense, b_diag, b_dense)

        def combine(lhs, rhs):
            Md_l, Mdense_l, bd_l, bdense_l = lhs
            Md_r, Mdense_r, bd_r, bdense_r = rhs

            Md = Md_r * Md_l
            bd = Md_r * bd_l + bd_r

            Mdense = torch.matmul(Mdense_r, Mdense_l)
            bdense = (
                torch.matmul(Mdense_r, bdense_l.unsqueeze(-1)).squeeze(-1) + bdense_r
            )

            return (Md, Mdense, bd, bdense)

        def apply(prefix, y0: torch.Tensor):
            Md, Mdense, bd, bdense = prefix

            y_diag0 = y0[:, :hdiag]
            y_dense0 = y0[:, hdiag:]

            y_diag = Md * y_diag0.unsqueeze(1) + bd

            y0d = y_dense0.unsqueeze(1).unsqueeze(-1)  # (B,1,b,1)
            y_dense = torch.matmul(Mdense, y0d).squeeze(-1) + bdense  # (B,C,b)

            return torch.cat([y_diag, y_dense], dim=-1)

        return combine, build, apply

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Run either recurrent or parallel chunked scan based on constructor settings.

        Args:
            X: (batch, seq, input_dim)
        """
        if not self.use_parallel:
            return self._forward_recurrent(X)

        return self._forward_parallel(X, chunk_size=self.chunk_size)

    # -------------------------
    # Recurrent implementation
    # -------------------------

    def _forward_recurrent(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, in_dim = X.shape

        inp = self._prepare_augmented_inputs(X)

        # Initialise the hidden state
        if self.input_dependent_init:
            y = self.init(X[:, 0, :])  # shape: (batch_size, hidden_dim)
        else:
            y = self.init.unsqueeze(0).expand(
                batch_size, -1
            )  # shape: (batch_size, hidden_dim)

        # Prepare a tensor to store all hidden states
        ys = torch.zeros(
            batch_size, seq_len, self.hidden_dim, device=X.device, dtype=X.dtype
        )

        # Recurrently compute the hidden states
        for i in range(seq_len):
            if self.diagonal_dense:
                y_diag = y[:, : -self.block_size]
                y_dense = y[:, -self.block_size :]
                diag_state_transition = self.vf_A_diag(inp[:, i]) * y_diag
                A = self.vf_A_dense(inp[:, i])
                dense_state_transition = torch.einsum(
                    "bij,bj->bi",
                    A.view(-1, self.block_size, self.block_size),
                    y_dense,
                )
                state_transition = torch.cat(
                    [diag_state_transition, dense_state_transition], dim=1
                )
            elif self.block_size > 1:
                state_transition = self.vf_A(inp[:, i]).view(
                    -1,
                    self.hidden_dim // self.block_size,
                    self.block_size,
                    self.block_size,
                )
                state_transition = (
                    state_transition
                    @ y.view(
                        -1,
                        self.hidden_dim // self.block_size,
                        self.block_size,
                        1,
                    )
                ).view(-1, self.hidden_dim)
            else:
                state_transition = self.vf_A(inp[:, i]) * y

            if self.bias:
                bias_term = self.vf_B(inp[:, i])
                state_transition += bias_term

            y = y + state_transition
            ys[:, i] = y

        return ys

    # -------------------------
    # Parallel chunked scan
    # -------------------------

    def _forward_parallel(self, X: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """
        Chunked parallel forward using torch.associative_scan (generic).

        Each step defines an affine transform:
            y_i = M_i y_{i-1} + b_i,  where M_i = I + A_i,  b_i = B_i
        We scan-combine transforms within each chunk, then apply prefixes
        to chunk-start state.
        """
        assoc_scan = torch._higher_order_ops.associative_scan

        batch_size, seq_len, _ = X.shape

        inp = self._prepare_augmented_inputs(X)

        if self.input_dependent_init:
            y_start = self.init(X[:, 0, :])
        else:
            y_start = self.init.unsqueeze(0).expand(batch_size, -1)

        ys = torch.empty(
            batch_size, seq_len, self.hidden_dim, device=X.device, dtype=X.dtype
        )

        if self.diagonal_dense:
            combine_fn, build_fn, apply_fn = self._scan_kernels_diagonal_dense()
        elif self.block_size > 1:
            combine_fn, build_fn, apply_fn = self._scan_kernels_blockdiag()
        else:
            combine_fn, build_fn, apply_fn = self._scan_kernels_elementwise()

        for start in range(0, seq_len, chunk_size):
            end = min(seq_len, start + chunk_size)
            inp_chunk = inp[:, start:end, :]  # (B, C, D+1)

            transforms = build_fn(
                inp_chunk
            )  # pytree of tensors with leading (B, C, ...)
            prefix_transforms = assoc_scan(
                combine_fn,
                transforms,
                dim=1,
                reverse=False,
                combine_mode="generic",
            )

            y_chunk = apply_fn(prefix_transforms, y_start)  # (B, C, H)
            ys[:, start:end, :] = y_chunk
            y_start = y_chunk[:, -1, :]

        return ys


class SLiCELayer(nn.Module):
    """
    A residual layer wrapping a SLiCE.

    SLiCELayer defaults to this structure:
      1. RMSNorm
      2. SLiCE
      3. Residual connection
      4. RMSNorm
      5. Token MLP with hidden size ff_mult * input_dim and GELU
      6. Residual connection
      7. Dropout on each residual branch

    Optional toggles for the LayerNorm + GLU/tanh single-stage wrapper:
      - norm_type="layernorm"
      - prenorm=False
      - ff_style="single"
      - ff_mult=1
      - ff_activation="glu" or "tanh"
      - dropout_position="output"

    The output dimension of the SLiCE is the same as the input dimension to preserve
    shape for the residual.

    Args:
        input_dim (int): Dimensionality of the input (and thus output) features.
        block_size (int): The size of the blocks along the diagonal of A in the SLiCE.
        diagonal_dense (bool): If True, A is composed of a diagonal matrix and a dense
                               block.
        init_std (float): Standard deviation for weight initialisation in the SLiCE.
        use_parallel (bool): Whether the inner SLiCE uses parallel scan execution.
        chunk_size (int): Chunk size used by the inner SLiCE when in parallel mode.
        dropout_rate (float): Dropout probability applied either on residual branches
                              or on the block output, depending on dropout_position.
        path_mode (str): How the inner SLiCE interprets the input path.
        norm_type (str): "rmsnorm" or "layernorm". Defaults to "rmsnorm".
        prenorm (bool): If True, apply normalisation before the SLiCE and
                        feedforward branches; if False, apply one norm after
                        both residual updates.
        ff_style (str): "mlp" for Linear -> activation -> Linear, or
                        "single" for a single Linear -> activation branch.
        ff_activation (str): "gelu", "glu", or "tanh".
        ff_mult (int): Expansion factor for the hidden feedforward size.
        dropout_position (str): "residual" to drop branch outputs before
                                residual addition, or "output" to drop the
                                final layer output.
        norm_eps (float): Epsilon used by the normalisation layers.

    Shape:
        - Input: (batch_size, seq_len, input_dim)
        - Output: (batch_size, seq_len, input_dim)
    """

    def __init__(
        self,
        input_dim: int,
        bias: bool = True,
        block_size: int = 4,
        diagonal_dense: bool = False,
        init_std: float = 0.01,
        scale: float = 1.0,
        input_dependent_init: bool = False,
        use_parallel: bool = True,
        chunk_size: int = 256,
        dropout_rate: float = 0.01,
        path_mode: str = "values",
        norm_type: str = "rmsnorm",
        prenorm: bool = True,
        ff_style: str = "mlp",
        ff_activation: str = "gelu",
        ff_mult: int = 4,
        dropout_position: str = "residual",
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if norm_type not in {"rmsnorm", "layernorm"}:
            raise ValueError("norm_type must be one of {'rmsnorm', 'layernorm'}.")
        if ff_style not in {"mlp", "single"}:
            raise ValueError("ff_style must be one of {'mlp', 'single'}.")
        if ff_activation not in {"gelu", "glu", "tanh"}:
            raise ValueError("ff_activation must be one of {'gelu', 'glu', 'tanh'}.")
        if ff_mult < 1:
            raise ValueError("ff_mult must be at least 1.")
        if ff_style == "single" and ff_mult != 1:
            raise ValueError("ff_mult must be 1 when ff_style='single'.")
        if dropout_position not in {"residual", "output"}:
            raise ValueError("dropout_position must be one of {'residual', 'output'}.")

        self.norm_type = norm_type
        self.prenorm = prenorm
        self.ff_style = ff_style
        self.ff_activation = ff_activation
        self.ff_mult = ff_mult
        self.dropout_position = dropout_position
        self.slice = SLiCE(
            input_dim=input_dim,
            hidden_dim=None,
            bias=bias,
            block_size=block_size,
            diagonal_dense=diagonal_dense,
            init_std=init_std,
            scale=scale,
            input_dependent_init=input_dependent_init,
            use_parallel=use_parallel,
            chunk_size=chunk_size,
            path_mode=path_mode,
        )

        self.drop = nn.Dropout(p=dropout_rate)
        if self.prenorm:
            if norm_type == "rmsnorm":
                self.slice_norm = RMSNorm(input_dim, eps=norm_eps)
                self.ff_norm = RMSNorm(input_dim, eps=norm_eps)
            else:
                self.slice_norm = nn.LayerNorm(input_dim, eps=norm_eps)
                self.ff_norm = nn.LayerNorm(input_dim, eps=norm_eps)
        else:
            if norm_type == "rmsnorm":
                self.norm = RMSNorm(input_dim, eps=norm_eps)
            else:
                self.norm = nn.LayerNorm(input_dim, eps=norm_eps)

        ff_hidden_dim = ff_mult * input_dim
        ff_in_dim = 2 * ff_hidden_dim if ff_activation == "glu" else ff_hidden_dim
        self.ff_in = nn.Linear(input_dim, ff_in_dim)
        self.ff_out = (
            nn.Linear(ff_hidden_dim, input_dim) if ff_style == "mlp" else nn.Identity()
        )
        if ff_activation == "gelu":
            self.act = nn.GELU()
        elif ff_activation == "glu":
            self.act = nn.GLU(dim=-1)
        else:
            self.act = nn.Tanh()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a configurable SLiCELayer.

        Args:
            X (torch.Tensor): shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: shape (batch_size, seq_len, input_dim)
        """
        slice_input = self.slice_norm(X) if self.prenorm else X
        slice_out = self.slice(slice_input)
        if self.dropout_position == "residual":
            X = X + self.drop(slice_out)
        else:
            X = X + slice_out

        ff_input = self.ff_norm(X) if self.prenorm else X
        ff_out = self.ff_out(self.act(self.ff_in(ff_input)))
        if self.dropout_position == "residual":
            X = X + self.drop(ff_out)
        else:
            X = X + ff_out

        if not self.prenorm:
            X = self.norm(X)
        if self.dropout_position == "output":
            X = self.drop(X)
        return X


class StackedSLiCE(nn.Module):
    """
    Stacks multiple SLiCELayers, preceded by an embedding layer and followed by a
    final linear layer.

    Args:
        num_layers (int): Number of SLiCELayers to stack.
        data_dim (int): Dimension of the input.
        hidden_dim (int): Hidden dimension used in each SLiCELayer.
        label_dim (int): Size of the output dimension.
        block_size (int): The size of the blocks along the diagonal of A in each layer.
        diagonal_dense (bool): If True, A is composed of a diagonal matrix and a dense
                               block in each layer.
        init_std (float): Standard deviation for the initialisation in each layer.
        use_parallel (bool): Whether each layer's inner SLiCE uses
                             parallel scan execution.
        chunk_size (int): Chunk size used by each layer's inner SLiCE in parallel mode.
        dropout_rate (float): Dropout probability applied in each layer.
        path_mode (str): How each inner SLiCE interprets its input path.
        norm_type (str): "rmsnorm" or "layernorm" for each stacked layer.
        prenorm (bool): Whether each stacked layer uses pre-norm.
        ff_style (str): "mlp" or "single" feedforward branch shape.
        ff_activation (str): "gelu", "glu", or "tanh".
        ff_mult (int): Expansion factor for the feedforward hidden size.
        dropout_position (str): "residual" or "output".
        norm_eps (float): Epsilon used by the normalisation layers.

    Shape:
        - Input: (batch_size, seq_len) if the input is tokens or
                 (batch_size, seq_len, data_dim) if the input is time-series values.
        - Output: (batch_size, seq_len, label_dim)
    """

    def __init__(
        self,
        num_layers: int,
        data_dim: int,
        hidden_dim: int,
        label_dim: int,
        bias: bool = True,
        tokens: bool = True,
        block_size: int = 4,
        diagonal_dense: bool = False,
        init_std: float = 0.01,
        scale: float = 1.0,
        input_dependent_init: bool = False,
        use_parallel: bool = True,
        chunk_size: int = 256,
        dropout_rate: float = 0.01,
        path_mode: str = "values",
        norm_type: str = "rmsnorm",
        prenorm: bool = True,
        ff_style: str = "mlp",
        ff_activation: str = "gelu",
        ff_mult: int = 4,
        dropout_position: str = "residual",
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.tokens = tokens
        if self.tokens:
            self.embedding = nn.Embedding(data_dim, hidden_dim)
        else:
            self.embedding = nn.Linear(data_dim, hidden_dim)

        # Build stacked SLiCE layers.
        self.layers = nn.ModuleList(
            [
                SLiCELayer(
                    input_dim=hidden_dim,
                    bias=bias,
                    block_size=block_size,
                    diagonal_dense=diagonal_dense,
                    init_std=init_std,
                    scale=scale,
                    input_dependent_init=input_dependent_init,
                    use_parallel=use_parallel,
                    chunk_size=chunk_size,
                    dropout_rate=dropout_rate,
                    path_mode=path_mode,
                    norm_type=norm_type,
                    prenorm=prenorm,
                    ff_style=ff_style,
                    ff_activation=ff_activation,
                    ff_mult=ff_mult,
                    dropout_position=dropout_position,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        # Final projection: from hidden_dim -> label_dim
        self.linear = nn.Linear(hidden_dim, label_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the stacked model:
            1. Embed input X
            2. Pass through each SLiCE layer
            3. Apply final linear projection

        Args:
            X (torch.Tensor): If tokens, shape (batch_size, seq_len)
                              If time-series, shape (batch_size, seq_len, data_dim)

        Returns:
            torch.Tensor: shape (batch_size, seq_len, label_dim)
        """
        # Step 1: Embedding
        if self.tokens:
            X = self.embedding(X.long())
        else:
            X = self.embedding(X.float())

        # Step 2: Pass through each stacked layer.
        for layer in self.layers:
            X = layer(X)  # (batch_size, seq_len, hidden_dim)

        # Step 3: Project to label_dim
        return self.linear(X)  # (batch_size, seq_len, label_dim)
