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
def add_outputs(input1: list, input2: list, weights: torch.Tensor | None = None):
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
                return [*[ipt * weights[0] for ipt in input1[:-1]], input1[-1] * weights[0] + input2 * weights[1]]
    elif isinstance(input1, torch.Tensor):
        if isinstance(input2, (list, tuple)):
            if weights is None:
                return [*input2[:-1], input1 + input2[-1]]
            else:
                assert len(weights) == 2
                return [*[ipt * weights[1] for ipt in input2[:-1]], input1 * weights[0] + input2[-1] * weights[1]]
        else:
            if weights is None:
                return input1 + input2
            else:
                assert len(weights) == 2
                return input1 * weights[0] + input2 * weights[1]
    else:
        raise NotImplementedError


def forward_parallel_net(flat_net: nn.Module,
                         seq_net: nn.Module,
                         x_past: torch.Tensor,
                         x_future: torch.Tensor,
                         forecast_only_flat: bool = True,
                         forecast_only_seq: bool = True,
                         seq_kwargs: dict = {},
                         flat_kwargs: dict = {},
                         out_weights: torch.Tensor | None = None):
    flat_out = flat_net(x_past, x_future, **flat_kwargs)
    seq_out = seq_net(x_past, x_future, **seq_kwargs)
    return add_outputs(flat_out, seq_out, out_weights)


def forward_concat_net(flat_net: nn.Module,
                       seq_net: nn.Module,
                       x_past: torch.Tensor,
                       x_future: torch.Tensor,
                       forecast_only_flat: bool = True,
                       forecast_only_seq: bool = True,
                       seq_kwargs: dict = {},
                       flat_kwargs: dict = {},
                       out_weights: torch.Tensor | None = None):
    flat_out = flat_net(x_past, x_future, **flat_kwargs, forward_only_with_net=True)
    backcast_flat_out, forecast_flat_out = flat_out
    if out_weights is None:
        x_past[:, :, :backcast_flat_out.shape[-1]] = backcast_flat_out
    else:
        n_targets = backcast_flat_out.shape[-1]
        x_past_targets = x_past[:, :, : n_targets]
        new_targets = backcast_flat_out[0] * out_weights[0] + x_past_targets * out_weights[1]
        x_past = torch.cat([new_targets, x_past[:, :, n_targets:]], dim=-1)

    x_future = torch.cat([forecast_flat_out, x_future], dim=-1)
    seq_out = seq_net(x_past, x_future, **seq_kwargs)
    if forecast_only_flat:
        flat_out = flat_out[1]
    if seq_net.forecast_only:
        return add_outputs(forecast_flat_out, seq_out, out_weights)
    else:
        return add_outputs(flat_out, seq_out, out_weights)
