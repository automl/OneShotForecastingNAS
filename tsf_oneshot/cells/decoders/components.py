import abc
import numpy as np

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import (
    PositionalEncoding
)
import torch
from torch import nn

from tsf_oneshot.cells.encoders.components import _Chomp1d, TCN_DEFAULT_KERNEL_SIZE
from tsf_oneshot.cells.utils import fold_tensor, unfold_tensor


class ForecastingDecoderLayer(nn.Module):
    @abc.abstractmethod
    def forward(self, x_future: torch.Tensor, encoder_output_layer: torch.Tensor, encoder_output_net: torch.Tensor,
                hx1: torch.Tensor, hx2: torch.Tensor, **kwargs):
        """
        A general function to fuse the information from the encoders. Given the different requirements from different
        network components, we will require the following inputs from the encoders
        :param x_future: future features, this is usually generated from the last decoder layer
        :param encoder_output_layer: encoder output from the same layer, this is required by the TCN decoders
        :param encoder_output_net: encoder output of from the network, this is  required by the transformer decoders
        :param hx1: hidden state from the same layer, this is required by the GRU/LSTM
        :param hx2: hidden state from the same layer, this is required by the LSTM network 
        :return:
        """
        raise NotImplementedError


class GRUDecoderModule(ForecastingDecoderLayer):
    def __init__(self, d_model: int, ts_skip_size: int = 1, bias: bool = True, dropout: float = 0.2, **kwargs):
        super(GRUDecoderModule, self).__init__()
        self.cell = nn.GRU(input_size=d_model, hidden_size=d_model, bias=bias, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ts_skip_size = ts_skip_size

    def forward(self, x_future: torch.Tensor, encoder_output_layer: torch.Tensor, encoder_output_net: torch.Tensor,
                hx1: torch.Tensor, hx2: torch.Tensor):
        if self.ts_skip_size > 1:
            x_future, size_info = fold_tensor(x_future, self.ts_skip_size)
            hx1 = hx1.repeat(1, self.ts_skip_size, 1)

        output, _ = self.cell(x_future, hx1)

        if self.ts_skip_size > 1:
            output = unfold_tensor(output, self.ts_skip_size, size_info)

        return self.dropout(output)


class LSTMDecoderModule(ForecastingDecoderLayer):
    def __init__(self, d_model: int, ts_skip_size: int = 1, bias: bool = True, dropout: float = 0.2, **kwargs):
        super(LSTMDecoderModule, self).__init__()
        self.cell = nn.LSTM(input_size=d_model, hidden_size=d_model, bias=bias, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ts_skip_size = ts_skip_size

    def forward(self, x_future: torch.Tensor, encoder_output_layer: torch.Tensor, encoder_output_net: torch.Tensor,
                hx1: torch.Tensor, hx2: torch.Tensor):
        if self.ts_skip_size > 1:
            x_future, size_info = fold_tensor(x_future, self.ts_skip_size)
            hx1 = hx1.repeat(1, self.ts_skip_size, 1)
            hx2 = hx2.repeat(1, self.ts_skip_size, 1)
        output, _ = self.cell(x_future, (hx1, hx2))

        if self.ts_skip_size > 1:
            output = unfold_tensor(output, self.ts_skip_size, size_info)
        return self.dropout(output)


class TransformerDecoderModule(ForecastingDecoderLayer):
    def __init__(self, d_model: int, forecasting_horizon: int,
                 nhead: int = 8, activation='gelu', dropout: float = 0.2, is_first_layer: bool = False,
                 ts_skip_size: int = 1, dim_feedforward: int | None = None,
                 **kwargs):
        super(TransformerDecoderModule, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.cell = nn.TransformerDecoderLayer(d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                               batch_first=True, activation=activation)
        self.is_first_layer = is_first_layer
        # we apply posititional encoding to all decoder inputs
        self.ts_skip_size = ts_skip_size
        if self.is_first_layer:
            self.ps_encoding = PositionalEncoding(d_model=d_model)
            W_pos = torch.empty((forecasting_horizon, d_model))
            self.dropout = nn.Dropout(dropout)
            nn.init.uniform_(W_pos, -0.02, 0.02)
            self.ps_encoding = nn.Parameter(W_pos, requires_grad=True)

    def forward(self, x_future: torch.Tensor, encoder_output_layer: torch.Tensor, encoder_output_net: torch.Tensor,
                hx1: torch.Tensor, hx2: torch.Tensor):
        if self.is_first_layer:
            x_future = self.dropout(self.ps_encoding + x_future)
        if self.ts_skip_size > 1:
            x_future, size_future = fold_tensor(x_future, skip_size=self.ts_skip_size)
            encoder_output_net, size_past = fold_tensor(encoder_output_net, self.ts_skip_size)
        mask = nn.Transformer.generate_square_subsequent_mask(x_future.shape[1], device=x_future.device)
        output = self.cell(x_future, memory=encoder_output_net, tgt_mask=mask,
                           )

        if self.ts_skip_size > 1:
            output = unfold_tensor(output, self.ts_skip_size, size_future)

        return output


class TCNDecoderModule(ForecastingDecoderLayer):
    def __init__(self, d_model: int,
                 kernel_size: int = TCN_DEFAULT_KERNEL_SIZE,
                 stride: int = 1, dilation: int = 1, dropout: float = 0.2, **kwargs):
        super(TCNDecoderModule, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        norm1 = nn.LayerNorm(d_model)
        chomp1 = _Chomp1d(padding)
        relu1 = nn.ReLU()
        dropout1 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, chomp1)
        self.net_post = nn.Sequential(norm1, relu1, dropout1)

        self.dropout = nn.Dropout(dropout)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * dilation

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)

    def forward(self, x_future: torch.Tensor, encoder_output_layer: torch.Tensor, encoder_output_net: torch.Tensor,
                hx1: torch.Tensor, hx2: torch.Tensor):
        # swap sequence and feature dimensions for use with convolutional nets
        # WE only need the feature maps that are within the receptive field
        len_x_future = x_future.shape[1]
        x_all = torch.cat([encoder_output_layer[:, -self.receptive_field - 1:], x_future], dim=1)
        x_all = x_all.transpose(1, 2).contiguous()

        out = self.net(x_all)[:, :, -len_x_future:]
        out = out.transpose(1, 2).contiguous()
        out = self.net_post(out)
        # out = out + x_future
        return out


class SepTCNDecoderModule(TCNDecoderModule):
    def __init__(self, d_model: int,
                 kernel_size: int = TCN_DEFAULT_KERNEL_SIZE,
                 stride: int = 1, dilation: int = 1, dropout: float = 0.2, **kwargs):
        super(TCNDecoderModule, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size,
                               stride=stride, padding=padding, dilation=dilation, groups=d_model)
        norm1 = nn.LayerNorm(d_model)
        linear = nn.Linear(d_model, d_model)
        chomp1 = _Chomp1d(padding)

        padding = (kernel_size - 1) * dilation
        relu2 = nn.ReLU()
        dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, chomp1)
        self.net_post = nn.Sequential(norm1, linear, relu2, dropout2)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * dilation

        self.init_weights()


class MLPMixDecoderModule(ForecastingDecoderLayer):
    # https://arxiv.org/pdf/2303.06053.pdf
    def __init__(self,
                 d_model: int,
                 window_size: int,
                 forecasting_horizon: int,
                 dropout: float = 0.2,
                 ts_skip_size: int = 1,
                 d_ff: int | None = None, **kwargs):
        super(MLPMixDecoderModule, self).__init__()
        if d_ff is None:
            d_ff = 2 * d_model
        self.forecasting_horizon = forecasting_horizon
        self.ts_skip_size = ts_skip_size

        n_past = int(np.ceil(window_size / self.ts_skip_size)) + int(np.ceil(forecasting_horizon / self.ts_skip_size))
        n_future = int(np.ceil((forecasting_horizon) / self.ts_skip_size))
        self.time_mixer = nn.Sequential(
            nn.Conv1d(n_past, n_future, 1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
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

    def forward(self, x_future: torch.Tensor, encoder_output_layer: torch.Tensor, encoder_output_net: torch.Tensor,
                hx1: torch.Tensor, hx2: torch.Tensor):
        if self.ts_skip_size > 1:
            encoder_output_net, size_past = fold_tensor(encoder_output_net, self.ts_skip_size)
            x_future, size_future = fold_tensor(x_future, self.ts_skip_size)

        input_t = torch.cat(
            [encoder_output_net, x_future], dim=1
        )

        out_t = self.time_norm(input_t)

        out_t = self.time_mixer(out_t)

        input_f = out_t + input_t[:, -x_future.shape[1]:, :]

        if self.ts_skip_size > 1:
            input_f = unfold_tensor(input_f, self.ts_skip_size, size_future)
        input_f_ = self.feature_norm(input_f)
        out_f = self.feature_mixer(input_f_)

        return out_f + input_f


class IdentityDecoderModule(ForecastingDecoderLayer):
    def __init__(self, d_model: int, **kwargse):
        super().__init__()

    def forward(self, x_future: torch.Tensor, encoder_output_layer: torch.Tensor, encoder_output_net: torch.Tensor,
                hx1: torch.Tensor, hx2: torch.Tensor):
        return x_future
