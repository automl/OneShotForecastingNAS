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
import torch.nn as nn
import torch.nn.functional as F

import types

from . import measure
from ..p_utils import get_layer_metric_array, reshape_elements


def fisher_forward_conv2d(self, x):
    x = F.conv2d(
        x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
    )
    # intercept and store the activations after passing through 'hooked' identity op
    self.act = self.dummy(x)
    return self.act


def fisher_forward_conv1d(self, x):
    x = F.conv1d(
        x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
    )
    # intercept and store the activations after passing through 'hooked' identity op
    self.act = self.dummy(x)
    return self.act


def fisher_forward_linear(self, x):
    x = F.linear(x, self.weight, self.bias)
    if hasattr(self, 'act') and isinstance(self.act, torch.Tensor):
        # Some of the linear layer are shared acorss different models. THis also needs to be checked
        self.act2 = self.dummy(x)
        return self.act2

    self.act = self.dummy(x)
    return self.act


def fisher_forward_gru(self, x, hx=None, ):
    x, hx = nn.GRU.forward(self, x, hx)

    self.act1 = self.dummy1(x)
    self.act2 = self.dummy2(hx)

    return self.act1, self.act2


def fisher_forward_lstm(self, x, hx=None, ):
    x, hx = nn.LSTM.forward(self, x, hx)
    self.act1 = self.dummy1(x)
    self.act2 = (self.dummy2(hx[0]), self.dummy2(hx[1]))
    return self.act1, self.act2


def get_layer_shape_array(net, metric, mode):
    metric_array = []
    for layer in net.modules():
        if mode == "channel" and hasattr(layer, "dont_ch_prune"):
            continue
        #if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        if isinstance(layer, (nn.Conv1d, nn.Linear)):
            metric_array.append(layer.weight.shape[1:])
        if isinstance(layer, (nn.LSTM, nn.GRU)):
            weight = layer.all_weights[0][0]
            metric_array.append(weight.shape[1:])

    return metric_array


@measure("fisher", bn=True, mode="channel")
def compute_fisher_per_weight(net, inputs_past, inputs_future, targets, loss_fn, mode, split_data=1):

    device = inputs_past.device

    if mode == "param":
        raise ValueError("Fisher pruning does not support parameter pruning.")

    net.train()
    all_hooks = []
    for n, layer in net.named_modules():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.GRU, nn.LSTM)):
            # variables/op needed for fisher computation
            layer.fisher = None
            layer.act = 0.0
            layer.dummy = nn.Identity()
            layer.name = n

            # replace forward method of conv/linear
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(fisher_forward_conv2d, layer)
            if isinstance(layer, nn.Conv1d):
                layer.forward = types.MethodType(fisher_forward_conv1d, layer)
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(fisher_forward_linear, layer)
            if isinstance(layer, nn.GRU):
                layer.forward = types.MethodType(fisher_forward_gru, layer)
                layer.dummy1 = nn.Identity()
                layer.dummy2 = nn.Identity()
            if isinstance(layer, nn.LSTM):
                layer.forward = types.MethodType(fisher_forward_lstm, layer)
                layer.dummy1 = nn.Identity()
                layer.dummy2 = nn.Identity()

            # function to call during backward pass (hooked on identity op at output of layer)
            def hook_factory(layer):
                def hook(module, grad_input, grad_output):
                    if isinstance(layer, (nn.GRU, nn.LSTM)):
                        if grad_output[0] is not None:
                            if not hasattr(layer, 'act1'):
                                return
                            act = layer.act1.detach()
                            grad = grad_output[0].detach()
                        elif grad_output[1] is not None:
                            if not hasattr(layer, 'act2'):
                                return
                            if isinstance(layer.act2, torch.Tensor):
                                act = layer.act2.detach()
                            else:
                                act = layer.act2[0].detach()
                            grad = grad_output[1].detach()
                        if len(act.shape) > 2:
                            g_nk = torch.sum((act * grad), [1, *list(range(3, len(act.shape)))])
                        else:
                            g_nk = act * grad
                        del_k = g_nk.pow(2).mean(0).mul(0.5)
                        if layer.fisher is None:
                            layer.fisher = del_k
                        else:
                            layer.fisher += del_k
                        if grad_output[0] is not None:
                            del (
                                layer.act1
                            )  # without deleting this, a nasty memory leak occurs! related: https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555
                        elif grad_output[1] is not None:
                            del (
                                layer.act2
                            )  # without deleting this, a nasty memory leak occurs! related: https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555

                    else:
                        if hasattr(layer, 'act'):
                            act = layer.act.detach()

                            grad = grad_output[0].detach()
                            if len(act.shape) > 2:
                                if isinstance(layer, nn.Conv1d):
                                    g_nk = torch.sum((act * grad), list(range(2, len(act.shape))))
                                else:
                                    g_nk = torch.sum((act * grad), [1, *list(range(3, len(act.shape)))])
                            else:
                                g_nk = act * grad
                            del_k = g_nk.pow(2).mean(0).mul(0.5)
                            if layer.fisher is None:
                                layer.fisher = del_k
                            else:
                                layer.fisher += del_k
                            del (
                                layer.act
                            )  # without deleting this, a nasty memory leak occurs! related: https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555
                        else:
                            act = layer.act2.detach()

                            grad = grad_output[0].detach()
                            if len(act.shape) > 2:
                                g_nk = torch.sum((act * grad), list(range(2, len(act.shape))))
                            else:
                                g_nk = act * grad
                            del_k = g_nk.pow(2).mean(0).mul(0.5)
                            if layer.fisher is None:
                                layer.fisher = del_k
                            else:
                                layer.fisher += del_k
                            del (
                                layer.act2
                            )  # without deleting this, a nasty memory leak occurs! related: https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555

                return hook

            # register backward hook on identity fcn to compute fisher info
            if isinstance(layer, (nn.LSTM, nn.GRU)):
                layer.dummy1.register_backward_hook(hook_factory(layer))
                layer.dummy2.register_backward_hook(hook_factory(layer))
            else:
                layer.dummy.register_backward_hook(hook_factory(layer))

    N = inputs_past.shape[0]
    net = net.float()
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        net.zero_grad()
        outputs = net(inputs_past[st:en].float(), inputs_future[st:en].float())
        loss = loss_fn(targets[st:en].float(), outputs)
        loss.backward()

    # retrieve fisher info
    def fisher(layer):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.GRU, nn.LSTM)):
            if layer.fisher is not None:
                return torch.abs(layer.fisher.detach())
            else:
                if isinstance(layer, (nn.LSTM, nn.GRU)):
                    weight = layer.all_weights[0][0]
                    return torch.zeros(weight.shape[0])  # size=ch
                else:
                    return torch.zeros(layer.weight.shape[0])  # size=ch
        else:
            return []

    grads_abs_ch = get_layer_metric_array(net, fisher, mode)

    # broadcast channel value here to all parameters in that channel
    # to be compatible with stuff downstream (which expects per-parameter metrics)
    # TODO cleanup on the selectors/apply_prune_mask side (?)

    def get_layer_shape(layer):
        #if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        if isinstance(layer, (nn.Conv1d, nn.Linear)):
            return layer.weight.shape[1:]
        if isinstance(layer, (nn.LSTM, nn.GRU)):
            weight = layer.all_weights[0][0]
            return weight.shape[1:]
        return []
        #return torch.zeros_like(layer.weight)

    shapes = get_layer_metric_array(net, get_layer_shape, mode)

    grads_abs = reshape_elements(grads_abs_ch, shapes, device)
    return grads_abs
