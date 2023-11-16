import abc
from typing import Tuple, Any
import torch
from torch import nn
from torch.nn.utils import weight_norm

from tsf_oneshot.cells.encoders.components import _Chomp1d


class ForecastingDecoderLayer(nn.Module):
    @abc.abstractmethod
    def forward(self, x_future: torch.Tensor, encoder_output_layer: torch.Tensor, encoder_output_net: torch.Tensor,
                hx1: torch.Tensor, hx2: torch.Tensor):
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
    def __init__(self, d_model: int, bias: bool = True):
        super(GRUDecoderModule, self).__init__()
        self.cell = nn.GRU(input_size=d_model, hidden_size=d_model, bias=bias, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_future: torch.Tensor, encoder_output_layer: torch.Tensor, encoder_output_net: torch.Tensor,
                hx1: torch.Tensor, hx2: torch.Tensor):
        output, _ = self.cell(x_future, hx1)
        return self.norm(output)


class LSTMDecoderModule(ForecastingDecoderLayer):
    def __init__(self, d_model: int, bias: bool = True):
        super(LSTMDecoderModule, self).__init__()
        self.cell = nn.LSTM(input_size=d_model, hidden_size=d_model, bias=bias, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_future: torch.Tensor, encoder_output_layer: torch.Tensor, encoder_output_net: torch.Tensor,
                hx1: torch.Tensor, hx2: torch.Tensor):
        output, _ = self.cell(x_future, (hx1, hx2))
        return output


class TransformerDecoderModule(ForecastingDecoderLayer):
    def __init__(self, d_model: int, nhead: int = 8):
        super(TransformerDecoderModule, self).__init__()
        self.cell = nn.TransformerDecoderLayer(d_model, nhead=nhead, dim_feedforward=4 * d_model, batch_first=True)
        self.hx_encoder_layer = nn.Linear(d_model, d_model)

    def forward(self, x_future: torch.Tensor, encoder_output_layer: torch.Tensor, encoder_output_net: torch.Tensor,
                hx1: torch.Tensor, hx2: torch.Tensor):
        mask = nn.Transformer.generate_square_subsequent_mask(x_future.shape[1], device=x_future.device)
        output = self.cell(x_future, memory=encoder_output_net, tgt_mask=mask)

        return output


class TCNDecoderModule(ForecastingDecoderLayer):
    def __init__(self, d_model: int, kernel_size: int = 7, stride: int = 1, dilation: int = 1, dropout: float = 0.1):
        super(TCNDecoderModule, self).__init__()
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
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)


        self.receptive_field = 1 + 2 * (kernel_size - 1) * dilation

    def forward(self,  x_future: torch.Tensor, encoder_output_layer: torch.Tensor, encoder_output_net: torch.Tensor,
                hx1: torch.Tensor, hx2: torch.Tensor):
        # swap sequence and feature dimensions for use with convolutional nets
        # WE only need the feature maps that are within the receptive field
        len_x_future = x_future.shape[1]
        x_all = torch.cat([encoder_output_layer[:, -self.receptive_field - 1:], x_future], dim=1)
        x_all = x_all.transpose(1, 2).contiguous()
        out = self.net(x_all)[:, :, -len_x_future:]
        out = out.transpose(1, 2).contiguous()
        out = self.relu(out + x_future)

        return self.norm(out)


class MLPDecoderModule(ForecastingDecoderLayer):
    def __init__(self,
                 d_model: int,
                 forecast_horizon: int,
                 d_bottleneck: int = 8,
                 n_liner_layers: int = 1,
                 d_model_linear: int = 1024,
                 ):
        super().__init__()
        self.forecasting_horizon = forecast_horizon
        self.d_model = d_model
        self.d_bottleneck = d_bottleneck

        act_func = nn.ReLU

        flatten_bottleneck_size = d_bottleneck * forecast_horizon
        networks_future = []
        networks_future.extend(
            [nn.Linear(d_model, d_bottleneck), act_func()])  # reduce the network to bottleneck dimensions
        networks_future.extend([nn.Flatten(1), nn.LayerNorm(flatten_bottleneck_size)])
        networks_future.extend([nn.Linear(flatten_bottleneck_size, d_model_linear), act_func(),
                                nn.LayerNorm(d_model_linear)])

        self.networks_future = nn.Sequential(*networks_future)

        network_linear = []
        d_start = d_model_linear + d_model_linear

        for i in range(n_liner_layers):
            network_linear.extend([nn.Linear(d_start, d_model_linear), act_func(), nn.LayerNorm(d_model_linear)])
            d_start = d_model_linear

        network_linear.extend(
            [nn.Linear(d_start, flatten_bottleneck_size), act_func(), nn.LayerNorm(flatten_bottleneck_size)])
        network_linear.extend(
            [nn.Unflatten(-1), [forecast_horizon, d_bottleneck],
             nn.Linear(d_bottleneck, d_model),
             act_func(),
             nn.LayerNorm(d_model)
             ]
        )
        self.network_linear = nn.Sequential(*network_linear)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_future: torch.Tensor, encoder_output_layer: torch.Tensor, encoder_output_net: torch.Tensor,
                hx1: torch.Tensor, hx2: torch.Tensor):
        encoder_output = encoder_output_net[:, -1]

        x_future = self.networks_future(x_future)

        x = torch.concat([encoder_output, x_future], dim=-1)
        x = self.global_layers(x)
        if self.local_layers is not None:
            x = self.local_layers(x)
        x = self.norm(x)
        return x


class IdentityDecoderModule(ForecastingDecoderLayer):
    def __init__(self, d_model: int):
        super().__init__()

    def forward(self, x_future: torch.Tensor, encoder_output_layer: torch.Tensor, encoder_output_net: torch.Tensor,
                hx1: torch.Tensor, hx2: torch.Tensor):
        return x_future
