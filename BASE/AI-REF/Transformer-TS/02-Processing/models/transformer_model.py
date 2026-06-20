# -*- coding: utf-8 -*-
"""
时序预测用 Transformer：输入 (B, seq_len, F)，输出 (B, pred_len, F)。
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        n_features,
        seq_len,
        pred_len,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_proj = nn.Linear(d_model, pred_len * n_features)

    def forward(self, x):
        # x: (B, seq_len, n_features)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.output_proj(x)
        return x.view(x.size(0), self.pred_len, self.n_features)
