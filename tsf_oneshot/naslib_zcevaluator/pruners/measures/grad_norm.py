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

import torch
from torch import nn
import torch.nn.functional as F

import copy

from . import measure
from ..p_utils import get_layer_metric_array


@measure("grad_norm", bn=True)
def get_grad_norm_arr(net, inputs_past, inputs_future, targets, loss_fn, split_data=1, skip_grad=False):
    net.zero_grad()
    N = inputs_past.shape[0]
    grad_norm_arr = []
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        outputs = net.forward(inputs_past[st:en], inputs_future[st:en])
        loss = loss_fn(targets[st:en], outputs)
        loss.backward()

        def get_layer_norm(layer):
            if isinstance(layer, (nn.Linear, nn.Conv1d)):
                return layer.weight.grad.norm() if layer.weight.grad is not None else torch.zeros_like(layer.weight)
            if isinstance(layer, (nn.GRU, nn.LSTM)):
                all_grad = []
                for weights in layer.all_weights:
                    for weight in weights:
                        if weight.grad is not None:
                            all_grad.append(weight.grad.norm())
                        else:
                            all_grad.append(torch.zeros_like(weight))
                return all_grad
            return layer.weight.grad.norm() if layer.weight.grad is not None else torch.zeros_like(layer.weight)
        grad_norm_arr.extend(get_layer_metric_array(
            net,
            get_layer_norm,
            mode="param",
        ))

    return grad_norm_arr
