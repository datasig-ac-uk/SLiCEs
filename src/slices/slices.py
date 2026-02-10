import warnings
from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn


class SLiCE(nn.Module):
    """
    A structured linear controlled differential equation (SLiCE) layer.

    Given a sequence of values (or increments) X_i in R^D for i=1,...,T, a SLiCE
    layer computes a sequence of hidden states y_i in R^H for i=1,...,T via the
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
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim
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

        # Add the increments of a sample counting channel.
        inc_ts = torch.full(
            (batch_size, seq_len, 1), 1.0, device=X.device, dtype=X.dtype
        )
        inp = torch.cat((inc_ts, X), dim=-1)  # shape: (batch_size, seq_len, x_dim)
        # Scale the input
        inp = inp * self.scale

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

        inc_ts = torch.full(
            (batch_size, seq_len, 1), 1.0, device=X.device, dtype=X.dtype
        )
        inp = torch.cat((inc_ts, X), dim=-1) * self.scale  # (B, T, D+1)

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


class SLiCEBlock(nn.Module):
    """
    A residual block wrapping a SLiCE. Includes:
      1. SLiCE forward pass
      2. Residual connection
      3. A Linear→GLU (or tanh) stage
      4. Residual connection
      5. LayerNorm
      6. Dropout

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
        dropout_rate (float): Dropout probability applied after the residual addition.
        use_glu (bool): Whether to apply a Linear -> GLU stage after the residual or
                            a Linear -> tanh stage.

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
        use_glu: bool = False,
    ):
        super().__init__()
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
        )
        self.norm = nn.LayerNorm(input_dim)

        # Linear -> GLU or Linear -> tanh stage
        self.use_glu = use_glu
        if self.use_glu:
            # Expand from input_dim -> 2*input_dim for GLU gating
            self.linear = nn.Linear(input_dim, 2 * input_dim)
            self.act = nn.GLU(dim=-1)
        else:
            self.linear = nn.Linear(input_dim, input_dim)
            self.act = lambda x: torch.tanh(x)

        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
            1. Compute SLiCE on input
            2. Apply residual skip connection
            3. Apply Linear -> GLU (or tanh) stage
            4. Add residual skip connection
            5. LayerNorm
            6. Dropout

        Args:
            X (torch.Tensor): shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: shape (batch_size, seq_len, input_dim)
        """
        # Step 1: SLiCE
        ys = self.slice(X)  # shape: (batch_size, seq_len, input_dim)

        # Step 2: Residual skip
        ys = ys + X

        # Step 3: Linear -> GLU (or tanh)
        ys_lin = self.linear(ys)  # shape: (batch_size, seq_len, 2*input_dim)
        ys_lin = self.act(ys_lin)  # shape: (batch_size, seq_len, input_dim)

        # Step 4: Residual skip
        ys = ys + ys_lin

        # Step 5: LayerNorm
        ys = self.norm(ys)

        # Step 6: Dropout
        ys = self.drop(ys)

        return ys


class StackedSLiCE(nn.Module):
    """
    Stacks multiple SLiCEBlocks, preceded by an embedding layer and followed by a
    final linear layer.

    Args:
        num_blocks (int): Number of SLiCEBlocks to stack.
        data_dim (int): Dimension of the input.
        hidden_dim (int): Hidden dimension used in each SLiCEBlock.
        label_dim (int): Size of the output dimension.
        block_size (int): The size of the blocks along the diagonal of A in each block.
        diagonal_dense (bool): If True, A is composed of a diagonal matrix and a dense
                               block in each block.
        init_std (float): Standard deviation for the initialisation in each block.
        use_parallel (bool): Whether each block's inner SLiCE uses
                             parallel scan execution.
        chunk_size (int): Chunk size used by each block's inner SLiCE in parallel mode.
        dropout_rate (float): Dropout probability applied in each block after the
                              residual.
        use_glu (bool): Whether to apply a Linear -> GLU or Linear -> tanh stage after
                        the residual.

    Shape:
        - Input: (batch_size, seq_len) if the input is tokens or
                 (batch_size, seq_len, data_dim) if the input is time-series values.
        - Output: (batch_size, seq_len, label_dim)
    """

    def __init__(
        self,
        num_blocks: int,
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
        use_glu: bool = False,
    ):
        super().__init__()
        self.tokens = tokens
        if self.tokens:
            self.embedding = nn.Embedding(data_dim, hidden_dim)
        else:
            self.embedding = nn.Linear(data_dim, hidden_dim)

        # Build the stack of SLiCE blocks
        self.blocks = nn.ModuleList(
            [
                SLiCEBlock(
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
                    use_glu=use_glu,
                )
                for _ in range(num_blocks)
            ]
        )

        # Final projection: from hidden_dim -> label_dim
        self.linear = nn.Linear(hidden_dim, label_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the stacked model:
            1. Embed input X
            2. Pass through each SLiCE block
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

        # Step 2: Pass through each SLiCE block
        for block in self.blocks:
            X = block(X)  # (batch_size, seq_len, hidden_dim)

        # Step 3: Project to label_dim
        return self.linear(X)  # (batch_size, seq_len, label_dim)
