import torch
from torch import nn

# pylint: disable=no-member


class LinearHead(nn.Module):
    """Head for text regression task. This code comes from transformers source code."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()

        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
