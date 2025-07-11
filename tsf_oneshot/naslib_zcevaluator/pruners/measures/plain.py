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

from . import measure
from ..p_utils import get_layer_metric_array


@measure("plain", bn=True, mode="param")
def compute_plain_per_weight(net, inputs_past, inputs_future, targets, mode, loss_fn, split_data=1):

    net.zero_grad()
    N = inputs_past.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        outputs = net.forward(inputs_past[st:en], inputs_future[st:en])
        loss = loss_fn( targets[st:en], outputs,)
        loss.backward()

    # select the gradients that we want to use for search/prune
    def plain(layer):
        if isinstance(layer, (nn.GRU, nn.LSTM)):
            plain_values = []
            for weights in layer.all_weights:
                for weight in weights:
                    if weight.grad is not None:
                        plain_values.append(weight.grad * weight)
                    else:
                        plain_values.append(torch.zeros_like(weight))
            return plain_values
        if layer.weight.grad is not None:
            return layer.weight.grad * layer.weight
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, plain, mode)
    return grads_abs
