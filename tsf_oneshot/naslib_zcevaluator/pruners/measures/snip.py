# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

from . import measure
from ..p_utils import get_layer_metric_array


@measure("snip", bn=True, mode="param")
def compute_snip_per_weight(net,  inputs_past, inputs_future, targets, mode, loss_fn, split_data=1):
    """
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)
    """
    # We directly compute the gradients within the snip function

    # Compute gradients (but don't apply them)
    net.zero_grad()
    N = inputs_past.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        outputs = net.forward(inputs_past[st:en], inputs_future[st:en])
        loss = loss_fn(targets[st:en], outputs)
        loss.backward()

    # select the gradients that we want to use for search/prune
    def snip(layer):
        if isinstance(layer, (nn.Linear, nn.Conv1d,)):
            mask_weight = layer.weight * layer.weight.grad if layer.weight.grad is not None else torch.zeros_like(layer.weight)
            return torch.abs(mask_weight)
        elif isinstance(layer, (nn.LSTM, nn.GRU)):
            all_mask_weights = []
            for weights in layer.all_weights:
                for weight in weights:
                    mask_weight = weight * weight.grad if weight.grad is not None else torch.zeros_like(weight)
                    all_mask_weights.append(torch.abs(mask_weight))
            return all_mask_weights
        else:
            raise ValueError(type(layer))

    grads_abs = get_layer_metric_array(net, snip, mode)

    return grads_abs
