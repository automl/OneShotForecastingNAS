from typing import Tuple, Any

import torch
from torch import nn
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.NBEATS_head import (
    get_trend_heads,
    get_generic_heads,
    get_seasonality_heads
)


class TSMLPBatchNormLayer(nn.Module):
    def __init__(self, n_dims: int, affine: bool=True):
        super(TSMLPBatchNormLayer, self).__init__()
        self.bn = nn.BatchNorm1d(n_dims, affine=affine)

    def forward(self, x):
        x = self.bn(torch.transpose(x, 1, 2)).transpose(1, 2)
        return x


class MLPFlatModule(nn.Module):
    def __init__(self, window_size: int, forecasting_horizon: int, dropout: float = 0.2,
                 is_last_layer: bool = False, norm_type='bn'):
        super(MLPFlatModule, self).__init__()
        if norm_type == 'ln':
            norm_layer = nn.LayerNorm(forecasting_horizon)
        elif norm_type == 'bn':
            norm_layer = TSMLPBatchNormLayer(forecasting_horizon)
        else:
            raise NotImplementedError
        if is_last_layer:
            self.net = nn.Linear(window_size + forecasting_horizon, forecasting_horizon)
        else:
            self.net = nn.Sequential(
                nn.Linear(window_size + forecasting_horizon, forecasting_horizon),
                norm_layer,
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        self.window_size = window_size
        self.forecasting_horizon = forecasting_horizon

    def forward(self, x_past: torch.Tensor, **kwargs):
        # here x_past should be the concatenation of x_past + x_future
        return torch.cat([x_past[:, :, :self.window_size], self.net(x_past)], -1)


class NBEATSModule(nn.Module):
    def __init__(self, window_size: int,
                 forecasting_horizon: int,
                 dropout: float = 0.2,
                 width=64, num_layers: int = 2, thetas_dim: int = 32,
                 stack_type: str = 't',
                 norm_type='bn'):
        super(NBEATSModule, self).__init__()
        fc_layers = []
        feature_in = window_size
        for _ in range(num_layers):
            if norm_type == 'ln':
                norm_layer = nn.LayerNorm(width)
            elif norm_type == 'bn':
                norm_layer = TSMLPBatchNormLayer(width)
            else:
                raise NotImplementedError

            fc_layers.extend(
                [
                    nn.Linear(feature_in, width, ),
                    norm_layer,
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            feature_in = width
        self.layers = nn.Sequential(*fc_layers)
        if stack_type == 't':
            backcast_head, forecast_head = get_trend_heads(block_width=width,
                                                           backcast_length=window_size,
                                                           forecast_length=forecasting_horizon,
                                                           thetas_dim=thetas_dim)
        elif stack_type == 's':
            backcast_head, forecast_head = get_seasonality_heads(block_width=width,
                                                                 backcast_length=window_size,
                                                                 forecast_length=forecasting_horizon,
                                                                 thetas_dim=thetas_dim)
        elif stack_type == 'g':
            backcast_head, forecast_head = get_generic_heads(block_width=width,
                                                             backcast_length=window_size,
                                                             forecast_length=forecasting_horizon,
                                                             thetas_dim=thetas_dim)
        else:
            raise NotImplementedError
        self.backcast_head = backcast_head
        self.forecast_head = forecast_head
        self.window_size = window_size
        self.forecasting_horizon = forecasting_horizon

    def forward(self, x_past: torch.Tensor, **kwargs):
        x_past_input = x_past.shape
        x = self.layers(x_past[:, :, :self.window_size]).flatten(0, 1)

        forecast = self.forecast_head(x)
        backcast = self.backcast_head(x)
        block_out = torch.cat([-backcast, forecast], dim=-1).view(x_past_input)
        # backcast = x_past - backcast
        # forecast = x_future + forecast
        return block_out + x_past


class IdentityFlatEncoderModule(nn.Module):
    def __init__(self, window_size: int, forecasting_horizon):
        super().__init__()

    def forward(self, x_past: torch.Tensor, **kwargs):
        return x_past
