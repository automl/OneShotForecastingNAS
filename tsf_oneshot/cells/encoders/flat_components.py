from typing import Tuple
import numpy as np

import torch
from torch.nn import functional as F
from torch import nn
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.NBEATS_head import (
    get_trend_heads,
    get_generic_heads,
    get_seasonality_heads
)

NBEATS_DEFAULT_THETA_DIMS = {'g': 128,
                             's': 8,
                             't': 4}


class TSMLPBatchNormLayer(nn.Module):
    def __init__(self, n_dims: int, affine: bool = True):
        super(TSMLPBatchNormLayer, self).__init__()
        self.bn = nn.BatchNorm1d(n_dims, affine=affine)

    def forward(self, x):
        x = self.bn(torch.transpose(x, 1, 2)).transpose(1, 2)
        return x


class _IdentityBasis(nn.Module):
    # https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/nhits.py
    def __init__(
            self,
            backcast_size: int,
            forecast_size: int,
            interpolation_mode: str,
            out_features: int = 1,
    ):
        super().__init__()
        assert (interpolation_mode in ["linear", "nearest"]) or (
                "cubic" in interpolation_mode
        )
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode
        self.out_features = out_features

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, :, : self.backcast_size]
        knots = theta[:, :, self.backcast_size:]

        # Interpolation is performed on default dim=-1 := H
        # knots = knots.reshape(len(knots), self.out_features, -1)
        if self.interpolation_mode in ["nearest", "linear"]:
            # knots = knots[:,None,:]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )
            # forecast = forecast[:,0,:]
        elif "cubic" in self.interpolation_mode:
            raise NotImplementedError
            if self.out_features > 1:
                raise Exception(
                    "Cubic interpolation not available with multiple outputs."
                )
            batch_size = len(backcast)
            knots = knots[:, None, :, :]
            forecast = torch.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size: (i + 1) * batch_size],
                    size=self.forecast_size,
                    mode="bicubic",
                )
                forecast[i * batch_size: (i + 1) * batch_size] += forecast_i[
                                                                  :, 0, 0, :
                                                                  ]  # [B,None,H,H] -> [B,H]
            forecast = forecast[:, None, :]  # [B,H] -> [B,None,H]
        else:
            raise NotImplementedError

        # [B,Q,H] -> [B,H,Q]
        # forecast = forecast.permute(0, 2, 1)
        return backcast, forecast


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

    def become_last_layer(self):
        self.net = nn.Linear(self.window_size + self.forecasting_horizon, self.forecasting_horizon)


class NBEATSModule(nn.Module):
    def __init__(self,
                 window_size: int,
                 forecasting_horizon: int,
                 dropout: float = 0.2,
                 width=256, num_fc_layers: int = 2, thetas_dim: int = 128,
                 stack_type: str = 't',
                 is_last_layer: bool = False,
                 has_fc_layers: bool = True,
                 norm_type='bn'):
        super(NBEATSModule, self).__init__()
        feature_in = window_size
        self.has_fc_layers = has_fc_layers

        self.window_size = window_size
        self.num_fc_layers = num_fc_layers
        self.forecasting_horizon = forecasting_horizon
        self.dropout = dropout
        self.width = width
        self.thetas_dim = thetas_dim
        self.norm_type = norm_type

        if has_fc_layers:
            self.fc_layers = self.generate_fc_layers(feature_in=feature_in, num_fc_layers=num_fc_layers,
                                                     width=width,
                                                     norm_type=norm_type,
                                                     dropout=dropout)
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

        self.is_last_layer = is_last_layer
        self.dropout_layer = nn.Dropout(dropout)
        if not is_last_layer:
            if norm_type == 'ln':
                self.norm_layer = nn.LayerNorm(window_size + forecasting_horizon)
            elif norm_type == 'bn':
                self.norm_layer = TSMLPBatchNormLayer(window_size + forecasting_horizon)
            else:
                raise NotImplementedError

    @staticmethod
    def generate_fc_layers(num_fc_layers: int, feature_in: int, width: int, norm_type: str = 'bn', dropout: float = 0.2,
                           **kwargs):
        fc_layers = []
        for i in range(num_fc_layers):
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
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    def forward(self, x_past: torch.Tensor, x_fc_output: torch.Tensor | None = None, **kwargs):
        x_past_input = x_past.shape
        if x_fc_output is None:
            x_fc_output = self.fc_layers(x_past[:, :, :self.window_size]).flatten(0, 1)
        else:
            x_fc_output = x_fc_output.flatten(0, 1)

        forecast = self.forecast_head(x_fc_output)
        backcast = self.backcast_head(x_fc_output)
        block_out = torch.cat([-backcast, forecast], dim=-1).view(x_past_input)
        return block_out + x_past
        #if not self.is_last_layer:
        #    return self.dropout_layer(self.norm_layer(block_out + x_past))
        #else:
        #    return block_out + x_past


class NHitsModule(nn.Module):
    # https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/nhits.py
    def __init__(self,
                 window_size: int,
                 forecasting_horizon: int,
                 dropout: float = 0.2,
                 width=64, num_layers: int = 2,
                 n_pool_kernel_size: int = 2,
                 n_freq_downsample: int = 4,
                 norm_type: str = 'bn',
                 interpolation_mode: str = 'linear',
                 is_last_layer: bool = False,
                 ):
        super(NHitsModule, self).__init__()
        self.window_size = window_size
        self.forecasting_horizon = forecasting_horizon

        feature_in = int(np.ceil(window_size / n_pool_kernel_size))
        n_theta = window_size + max(forecasting_horizon // n_freq_downsample, 1)
        fc_layers = []
        self.pooling_layer = nn.MaxPool1d(
            kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True
        )

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
        fc_layers.append(nn.Linear(feature_in, n_theta))

        self.layers = nn.Sequential(*fc_layers)
        self.basic = _IdentityBasis(backcast_size=window_size,
                                    forecast_size=forecasting_horizon,
                                    out_features=1,
                                    interpolation_mode=interpolation_mode)

    def forward(self, x_past: torch.Tensor, **kwargs):
        theta = self.layers(self.pooling_layer(x_past[:, :, :self.window_size]))
        backcast, forecast = self.basic(theta)
        block_out = torch.cat([-backcast, forecast], dim=-1)
        return block_out + x_past


class IdentityFlatEncoderModule(nn.Module):
    def __init__(self, window_size: int, forecasting_horizon, **kwargs):
        super().__init__()

    def forward(self, x_past: torch.Tensor, **kwargs):
        return x_past
