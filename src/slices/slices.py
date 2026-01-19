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
        init_std: float = 1.0,
        scale: float = 1.0 / 40,
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

        if diagonal_dense and self.block_size == self.hidden_dim:
            self.diagonal_dense = (
                False  # No point in diagonal + dense if only one block
            )
        elif diagonal_dense and self.block_size == 1:
            self.diagonal_dense = False  # No point in diagonal + dense if no dense part
        else:
            self.diagonal_dense = diagonal_dense

        # Define learnt initial hidden state y_0
        self.init = torch.nn.Parameter(torch.randn(self.hidden_dim) * self.init_std)

        if self.diagonal_dense:
            # For diagonal + dense block structure, define separate parameters
            # for the diagonal and dense parts.
            self.vf_A_diag = nn.Linear(
                self.input_dim + 2, self.hidden_dim - self.block_size, bias=False
            )
            self.vf_A_dense = nn.Linear(
                self.input_dim + 2, self.block_size * self.block_size, bias=False
            )
            nn.init.normal_(self.vf_A_diag.weight, mean=0.0, std=self.init_std)
            nn.init.normal_(self.vf_A_dense.weight, mean=0.0, std=self.init_std)
        else:
            # Define the vector field A as a linear layer
            self.vf_A = nn.Linear(
                self.input_dim + 2, self.hidden_dim * self.block_size, bias=False
            )
            nn.init.normal_(self.vf_A.weight, mean=0.0, std=self.init_std)

        if bias:
            self.vf_B = nn.Linear(self.input_dim + 2, self.hidden_dim, bias=False)
            nn.init.normal_(self.vf_B.weight, mean=0.0, std=self.init_std)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
            Recurrently computes the hidden states y_i given inputs X_i.

        Args:
            X (torch.Tensor): shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, in_dim = X.shape

        # Add the value and increments of a sample counting channel.
        inc_ts = torch.full(
            (batch_size, seq_len, 1), 1.0, device=X.device, dtype=X.dtype
        )
        ts = torch.cumsum(inc_ts, dim=1)  # shape: (batch_size, seq_len, 1)
        inp = torch.cat((inc_ts, ts, X), dim=-1)  # shape: (batch_size, seq_len, x_dim)

        # Scale inputs
        inp = inp * self.scale

        # Initialise the hidden state
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
        init_std: float = 1.0,
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
        init_std: float = 1.0,
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
