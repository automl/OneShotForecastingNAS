import inspect

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


# this function is applied to integrate the output of different networks
def add_outputs(input1: list, input2: list):
    if isinstance(input1, (list, tuple)):
        if isinstance(input2, (list, tuple)):
            assert len(input2) == len(input1)
            res = [None for _ in range(len(input1))]
            for i, (i1, i2) in enumerate(zip(input1, input2)):
                res[i] = add_outputs(i1, i2)
            return res
        else:
            assert isinstance(input2, torch.Tensor)
            return [*input1[:-1], input1[-1] + input2]
    elif isinstance(input1, torch.Tensor):
        if isinstance(input2, (list, tuple)):
            return [*input2[:-1], input1 + input2[-1]]
        else:
            return input1 + input2
    else:
        raise NotImplementedError


def forward_parallel_net(flat_net: nn.Module,
                         seq_net: nn.Module,
                         x_past: torch.Tensor,
                         x_future: torch.Tensor,
                         forecast_only_flat: bool = True,
                         forecast_only_seq: bool = True,
                         seq_kwargs: dict = {},
                         flat_kwargs: dict = {}):
    flat_out = flat_net(x_past, x_future, **flat_kwargs)
    seq_out = seq_net(x_past, x_future, **seq_kwargs)
    return add_outputs(flat_out, seq_out)


def forward_concat_net(flat_net: nn.Module,
                       seq_net: nn.Module,
                       x_past: torch.Tensor,
                       x_future: torch.Tensor,
                       forecast_only_flat: bool = True,
                       forecast_only_seq: bool = True,
                       seq_kwargs: dict = {},
                       flat_kwargs: dict = {}):
    flat_out = flat_net(x_past, x_future, **flat_kwargs, forward_only_with_net=True)
    backcast_flat_out, forecast_flat_out = flat_out
    # x_past[:,:, :self.d_output] = backcast_flat_out
    x_future = torch.cat([forecast_flat_out, x_future], dim=-1)
    seq_out = seq_net(x_past, x_future, **seq_kwargs)
    if forecast_only_flat:
        flat_out = flat_out[1]
    if seq_net.forecast_only:
        return add_outputs(forecast_flat_out, seq_out)
    else:
        return add_outputs(flat_out, seq_out)
