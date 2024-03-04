from typing import Tuple, Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import weight_norm

from tsf_oneshot.cells.encoders.flat_components import TSMLPBatchNormLayer
from tsf_oneshot.cells.utils import fold_tensor, unfold_tensor, _Chomp1d

TCN_DEFAULT_KERNEL_SIZE = 15


class GRUEncoderModule(nn.Module):
    def __init__(self, d_model: int, ts_skip_size: int = 1, bias: bool = False,
                 bidirectional: bool = True, dropout: float = 0.2, **kwargs):
        super(GRUEncoderModule, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        d_input = d_model
        self.hx_encoder_layer = nn.Linear(d_model, d_model)

        if bidirectional:
            assert d_model % 2 == 0
            d_model = d_model // 2

        self.cell = nn.GRU(input_size=d_input, hidden_size=d_model, bias=bias, num_layers=1,
                           batch_first=True, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.ts_skip_size = ts_skip_size

    def forward(self, x_past: torch.Tensor, hx: Tuple[torch.Tensor] | None = None):
        if self.ts_skip_size > 1:
            x_past, size_info = fold_tensor(x_past, self.ts_skip_size)

        output, hx = self.cell(x_past, hx)

        if self.bidirectional:
            # we ask the output to be
            output = torch.cat([output[:, :, :self.d_model], torch.flip(output[:, :, :self.d_model], dims=(1,))], -1)
            hx = torch.cat([hx[:1, :, :], torch.flip(hx[1:, :, :], dims=(1,))], -1)

        if self.ts_skip_size > 1:
            output = unfold_tensor(output, self.ts_skip_size, size_info)
            hx = hx[:, :size_info[0], :]

        return self.dropout(output), hx, self.hx_encoder_layer(hx)


class LSTMEncoderModule(nn.Module):
    def __init__(self, d_model: int, ts_skip_size: int = 1, bias: bool = True,
                 bidirectional: bool = False, dropout=0.2,
                 **kwargs):
        super(LSTMEncoderModule, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        d_input = d_model
        if bidirectional:
            assert d_model % 2 == 0
            d_model = d_model // 2

        self.cell = nn.LSTM(input_size=d_input, hidden_size=d_model, bias=bias, num_layers=1, batch_first=True,
                            bidirectional=bidirectional)
        self.bidirectional = bidirectional
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.ts_skip_size = ts_skip_size

    def forward(self, x_past: torch.Tensor, hx: Tuple[torch.Tensor] | None = None):
        if self.ts_skip_size > 1:
            x_past, size_info = fold_tensor(x_past, self.ts_skip_size)

        output, hx = self.cell(x_past, hx)
        if self.bidirectional:
            output = torch.cat([output[:, :, :self.d_model], torch.flip(output[:, :, :self.d_model], dims=(1,))], -1)
            hx = (
                torch.cat([hx[0][:1, :, :], torch.flip(hx[0][1:, :, :], dims=(1,))], -1),
                torch.cat([hx[1][:1, :, :], torch.flip(hx[1][1:, :, :], dims=(1,))], -1),
            )

        if self.ts_skip_size > 1:
            output = unfold_tensor(output, self.ts_skip_size, size_info)
            hx = (hx[0][:, :size_info[0], :], hx[1][:, :size_info[0], :])
        return self.dropout(output), *hx


class TransformerEncoderModule(nn.Module):
    def __init__(self, d_model: int, window_size: int, nhead: int = 8, activation='gelu', dropout: float = 0.2,
                 is_casual_model: bool = False, is_first_layer: bool = False, ts_skip_size: int = 1, **kwargs):
        super(TransformerEncoderModule, self).__init__()
        self.cell = nn.TransformerEncoderLayer(d_model, nhead=nhead, dim_feedforward=4 * d_model, batch_first=True,
                                               dropout=dropout,
                                               activation=activation)
        self.is_casual_model: bool = is_casual_model
        self.hx_encoder_layer = nn.Linear(d_model, d_model)
        self.is_first_layer = is_first_layer
        self.ts_skip_size = ts_skip_size
        if self.is_first_layer:
            W_pos = torch.empty((window_size, d_model))
            # https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/layers/PatchTST_layers.py#L96
            self.dropout = nn.Dropout(dropout)
            nn.init.uniform_(W_pos, -0.02, 0.02)
            self.ps_encoding = nn.Parameter(W_pos, requires_grad=True)

    def forward(self, x_past: torch.Tensor, hx: Any | None = None):
        if self.is_first_layer:
            # x_past = self.ps_encoding(x_past)
            x_past = self.dropout(self.ps_encoding + x_past)
        if self.ts_skip_size > 1:
            x_past, size_info = fold_tensor(x_past, self.ts_skip_size)

        if self.is_casual_model:
            mask = nn.Transformer.generate_square_subsequent_mask(x_past.shape[1], device=x_past.device)
        else:
            mask = None
        output = self.cell(x_past, src_mask=mask)
        # Hidden States
        if self.ts_skip_size > 1:
            output = unfold_tensor(output, self.ts_skip_size, size_info)
        hidden_states = output[:, [-1]].transpose(0, 1)
        return output, hidden_states, self.hx_encoder_layer(hidden_states)


class TCNEncoderModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = TCN_DEFAULT_KERNEL_SIZE, stride: int = 1, dilation: int = 1,
                 dropout: float = 0.2, **kwargs):
        super(TCNEncoderModule, self).__init__()
        # dilation = ts_skip_size
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(d_model, d_model, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.linear = nn.Conv1d(d_model, d_model, 1)
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(d_model, d_model, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

        self.hx_encoder_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x_past: torch.Tensor, hx: Any | None = None):
        # swap sequence and feature dimensions for use with convolutional nets
        x_past = x_past.transpose(1, 2).contiguous()
        out = self.net(x_past)
        out = out + x_past
        out = out.transpose(1, 2).contiguous()
        hx = out[:, [-1]].transpose(0, 1)
        return self.dropout(out), hx, self.hx_encoder_layer(hx)


class MLPMixEncoderModule(nn.Module):
    # https://arxiv.org/pdf/2303.06053.pdf
    # https://github.com/google-research/google-research/blob/master/tsmixer/tsmixer_basic/models/tsmixer.py
    def __init__(self, d_model: int, window_size: int, dropout: float = 0.2,
                 forecasting_horizon: int = 0, d_ff: int | None = None, ts_skip_size: int = 1, **kwargs):
        super(MLPMixEncoderModule, self).__init__()
        self.ts_skip_size = ts_skip_size
        series_len = int(np.ceil(window_size / ts_skip_size))
        self.time_mixer = nn.Sequential(
            nn.Linear(series_len, series_len),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # self.time_norm = nn.BatchNorm1d(d_model * window_size)
        self.time_norm = nn.LayerNorm(d_model)

        if d_ff is None:
            d_ff = d_model * 2

        self.feature_mixer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.feature_norm = nn.LayerNorm(d_model)
        self.hx_encoder_layer = nn.Linear(d_model, d_model)

    def forward(self, x_past: torch.Tensor, hx: Any | None = None):
        x_past, size_info = fold_tensor(x_past, self.ts_skip_size)
        input_t = self.time_norm(x_past).transpose(1, 2).contiguous()
        out_t = self.time_mixer(input_t)

        input_f = out_t.transpose(1, 2).contiguous()

        input_f += input_f + x_past

        input_f = unfold_tensor(input_f, self.ts_skip_size, size_info)
        input_f_shape = input_f.shape
        input_f = self.feature_norm(input_f)
        out_f = self.feature_mixer(input_f)

        out = out_f + input_f
        hx = out[:, [-1]].transpose(0, 1)

        return out, hx, self.hx_encoder_layer(hx)


class IdentityEncoderModule(nn.Module):
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.hx_encoder_layer = nn.Linear(d_model, d_model)

    def forward(self, x_past: torch.Tensor, hx: Any | None = None):
        hx = x_past[:, [-1]].transpose(0, 1)
        return x_past, hx, self.hx_encoder_layer(hx)
