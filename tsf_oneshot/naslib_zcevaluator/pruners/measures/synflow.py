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

from . import measure
from ..p_utils import get_layer_metric_array


@measure("synflow", bn=False, mode="param")
@measure("synflow_bn", bn=True, mode="param")
def compute_synflow_per_weight(net, inputs_past, inputs_future, targets, mode, split_data=1, loss_fn=None):

    device = inputs_past.device

    # convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if "weight_mask" not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net.double())

    # Compute gradients with input of 1s
    net.zero_grad()
    net.double()
    input_past_dim = list(inputs_past[0, :].shape)
    input_future_dim = list(inputs_future[0,:].shape)
    # inputs = torch.ones([1] + input_dim).to(device)
    inputs_past = torch.ones([1] + input_past_dim).double().to(device)
    inputs_future = torch.ones([1] + input_future_dim).double().to(device)

    output = net.forward(inputs_past, inputs_future)
    if isinstance(output, list):
        torch.sum(output[1]).backward()
    else:
        torch.sum(output).backward()

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if isinstance(layer, (nn.LSTM, nn.GRU)):
            all_synflow_values = []
            for weights in layer.all_weights:
                for weight in weights:
                    if weight.grad is not None:
                        all_synflow_values.append(torch.abs(weight.grad * weight))
                    else:
                        all_synflow_values.append(torch.zeros_like(weight))
            return all_synflow_values
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, synflow, mode)

    # apply signs of all params
    nonlinearize(net, signs)

    return grads_abs
