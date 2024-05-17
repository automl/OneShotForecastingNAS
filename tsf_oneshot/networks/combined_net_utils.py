import inspect
from  typing import Callable

import torch
from torch import nn


def get_kwargs():
    # https://stackoverflow.com/a/65927265
    # get the values and arguments of the current function
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != 'self':
            kwargs[key] = values[key]
    return kwargs


# this function is applied to integrate the output of different networks.
def add_outputs(input1: list | torch.Tensor, input2: list | torch.Tensor, weights: torch.Tensor | None = None):
    if isinstance(input1, (list, tuple)):
        if isinstance(input2, (list, tuple)):
            assert len(input2) == len(input1)
            res = [None for _ in range(len(input1))]
            for i, (i1, i2) in enumerate(zip(input1, input2)):
                res[i] = add_outputs(i1, i2, weights)
            return res
        else:
            assert isinstance(input2, torch.Tensor)
            if weights is None:
                return [*input1[:-1], input1[-1] + input2]
            else:
                assert len(weights) == 2
                return [add_outputs(ipt, input2, weights) for ipt in input1]
    elif isinstance(input1, torch.Tensor):
        if isinstance(input2, (list, tuple)):
            if weights is None:
                return [*input2[:-1], input1 + input2[-1]]
            else:
                return [add_outputs(input1, ipt, weights) for ipt in input2]
        else:
            if weights is None:
                return input1 + input2
            else:
                assert len(weights) == 2
                return input1 * weights[0] + input2 * weights[1]
    else:
        raise NotImplementedError


def decompose_input_variables(x: torch.Tensor, n_vars:int, decompose: Callable) -> list[torch.Tensor]:
    x_variables = x[:, :, :n_vars]
    seasonal, trend = decompose(x_variables)

    x = [
        torch.cat([trend, x[:, :, n_vars:]], -1),
        torch.cat([seasonal, x[:, :, n_vars:]], -1),
    ]
    return x


def forward_parallel_net(flat_net: nn.Module,
                         seq_net: nn.Module,
                         x_past: torch.Tensor,
                         x_future: torch.Tensor,
                         decompose,
                         forecast_only_flat: bool = True,
                         forecast_only_seq: bool = True,
                         seq_kwargs: dict = {},
                         flat_kwargs: dict = {},
                         out_weights: torch.Tensor | None = None):

    flat_out = flat_net(x_past, x_future, **flat_kwargs)

    n_vars = flat_out[-1].shape[-1]
    x_past = decompose_input_variables(x_past, n_vars, decompose)

    seq_out = seq_net(x_past, x_future, **seq_kwargs)

    if forecast_only_flat:
        flat_out = flat_out[1]
    if forecast_only_seq:
        seq_out = seq_out[1]
    return add_outputs(flat_out, seq_out, out_weights)


def forward_concat_net(flat_net: nn.Module,
                       seq_net: nn.Module,
                       x_past: torch.Tensor,
                       x_future: torch.Tensor,
                       decompose,
                       forecast_only_flat: bool = True,
                       forecast_only_seq: bool = True,
                       seq_kwargs: dict = {},
                       flat_kwargs: dict = {},
                       out_weights: torch.Tensor | None = None):
    flat_out = flat_net(x_past, x_future, **flat_kwargs, forward_only_with_net=True)
    backcast_flat_out, forecast_flat_out = flat_out

    # x_past contains two values, the first one is from the raw data, the second one is from the decoder architecture
    # HERE we have x past as the first item and backcast_flat_out as the second input
    n_vars = backcast_flat_out.shape[-1]

    x_past = decompose_input_variables(x_past, n_vars, decompose)

    seasonal_future, trend_future = decompose(forecast_flat_out)

    x_future = [
        torch.cat([trend_future, x_future], dim=-1),
        torch.cat([seasonal_future, x_future], dim=-1)
    ]

    seq_out = seq_net(x_past, x_future, **seq_kwargs)

    if forecast_only_flat:
        flat_out = flat_out[1]
    if seq_net.forecast_only:
        return add_outputs(forecast_flat_out, seq_out, out_weights)
    else:
        return add_outputs(flat_out, seq_out, out_weights)
