from typing import Tuple, Any

import torch
from torch import nn
from torch.nn.utils import weight_norm

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import \
    PositionalEncoding

TCN_DEFAULT_KERNEL_SIZE=15


class GRUEncoderModule(nn.Module):
    def __init__(self, d_model: int, bias: bool = True):
        super(GRUEncoderModule, self).__init__()
        self.cell = nn.GRU(input_size=d_model, hidden_size=d_model, bias=bias, num_layers=1, batch_first=True)
        self.hx_encoder_layer = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_past: torch.Tensor, hx: Tuple[torch.Tensor] | None = None):
        output, hx = self.cell(x_past, hx)

        return self.norm(output), hx, self.hx_encoder_layer(hx)


class LSTMEncoderModule(nn.Module):
    def __init__(self, d_model: int, bias: bool = True):
        super(LSTMEncoderModule, self).__init__()
        self.cell = nn.LSTM(input_size=d_model, hidden_size=d_model, bias=bias, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_past: torch.Tensor, hx: Tuple[torch.Tensor] | None = None):
        output, hx = self.cell(x_past, hx)
        return self.norm(output), *hx


class TransformerEncoderModule(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, activation='gelu', dropout:float=0.2, is_casual_model: bool = False, is_first_layer: bool=False):
        super(TransformerEncoderModule, self).__init__()
        self.cell = nn.TransformerEncoderLayer(d_model, nhead=nhead, dim_feedforward=4 * d_model, batch_first=True,
                                               dropout=dropout,
                                               activation=activation)
        self.is_casual_model: bool = is_casual_model
        self.hx_encoder_layer = nn.Linear(d_model, d_model)
        self.is_first_layer = is_first_layer
        if self.is_first_layer:
            self.ps_encoding = PositionalEncoding(d_model=d_model)

    def forward(self, x_past: torch.Tensor, hx: Any | None = None):
        if self.is_casual_model:
            mask = nn.Transformer.generate_square_subsequent_mask(x_past.shape[0], device=x_past.device)
        else:
            mask = None
        if self.is_first_layer:
            x_past = self.ps_encoding(x_past)
        output = self.cell(x_past, src_mask=mask)
        # Hidden States
        hidden_states = output[:, [-1]].transpose(0, 1)

        return output, hidden_states, self.hx_encoder_layer(hidden_states)


# _Chomp1d, _TemporalBlock and _TemporalConvNet original implemented by
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py, Carnegie Mellon University Locus Labs
# Paper: https://arxiv.org/pdf/1803.01271.pdf
class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super(_Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TCNEncoderModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = TCN_DEFAULT_KERNEL_SIZE, stride: int = 1, dilation: int = 1, dropout: float = 0.2):
        super(TCNEncoderModule, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(d_model, d_model, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(d_model, d_model, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2,)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

        self.hx_encoder_layer = nn.Linear(d_model, d_model)

    def forward(self, x_past: torch.Tensor, hx: Any | None = None):
        # swap sequence and feature dimensions for use with convolutional nets
        x_past = x_past.transpose(1, 2).contiguous()

        out = self.net(x_past)
        out = self.relu(out + x_past)
        out = out.transpose(1, 2).contiguous()
        out = self.norm(out)

        hx = out[:, [-1]].transpose(0, 1)
        return out, hx, self.hx_encoder_layer(hx)


class MLPMixEncoderModule(nn.Module):
    # https://arxiv.org/pdf/2303.06053.pdf
    def __init__(self, d_model: int, window_size: int, dropout: float = 0.2, forecasting_horizon: int=0):
        super(MLPMixEncoderModule, self).__init__()
        self.time_mixer = nn.Sequential(
            nn.Linear(window_size, window_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.time_norm = nn.LayerNorm(window_size)

        self.feature_mixer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.feature_norm = nn.LayerNorm(d_model)
        self.hx_encoder_layer = nn.Linear(d_model, d_model)

    def forward(self, x_past: torch.Tensor, hx: Any | None = None):
        input_t = x_past.transpose(1, 2).contiguous()
        out_t = self.time_mixer(input_t)
        out_t = self.time_norm(out_t + input_t)

        input_f = out_t.transpose(1, 2).contiguous()
        out_f = self.feature_mixer(input_f)
        out = self.feature_norm(input_f + out_f)
        hx = out[:, [-1]].transpose(0, 1)
        return out, hx, self.hx_encoder_layer(hx)


class IdentityEncoderModule(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.hx_encoder_layer = nn.Linear(d_model, d_model)

    def forward(self, x_past: torch.Tensor, hx: Any | None = None):
        hx = x_past[:, [-1]].transpose(0, 1)
        return x_past, hx, self.hx_encoder_layer(hx)
