import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dim_feedforward=256,
        dropout=0.1
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Dropouts
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        # BatchNorm (over feature dim)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)

    def forward(self, x, causal_mask=None, padding_mask=None):
        """
        x: (B, T, D)
        """

        # ---- Self Attention ----
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask
        )

        x = x + self.dropout_attn(attn_out)

        # BatchNorm expects (B, D, T)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)

        # ---- Feedforward ----
        ffn_out = self.linear2(
            self.dropout_ffn(
                F.relu(self.linear1(x))
            )
        )

        x = x + ffn_out

        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)

        return x


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dim_feedforward=256,
        dropout=0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

    def forward(self, x, padding_mask=None):
        """
        x: (B, T, D)
        padding_mask: (B, T)  True for padded positions
        """

        B, T, D = x.shape

        # ---- causal mask (shared across layers) ----
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device),
            diagonal=1
        ).bool()

        for layer in self.layers:
            x = layer(
                x,
                causal_mask=causal_mask,
                padding_mask=padding_mask
            )

        return x   # (B, T, D)
