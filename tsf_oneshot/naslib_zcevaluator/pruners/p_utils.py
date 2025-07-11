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

def get_some_data(train_dataloader, preprocessor, num_batches, device):
    inputs_past = []
    inputs_future = []
    targets = []
    dataloader_iter = iter(train_dataloader)
    for _ in range(num_batches):
        X, y = next(dataloader_iter)
        x_past, x_future, (loc, scale) = preprocessor.preprocessing(X)

        inputs_past.append(x_past)
        inputs_future.append(x_future)
        targets.append(y['future_targets'])

    inputs_past = torch.cat(inputs_past)
    inputs_future = torch.cat(inputs_future)
    targets = torch.cat(targets)

    inputs_past = inputs_past.to(device)
    inputs_future = inputs_future.to(device)
    targets = targets.to(device)
    return inputs_past, inputs_future, targets.float()


def get_some_data_grasp(train_dataloader, num_classes, samples_per_class, device):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(train_dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx : idx + 1], targets[idx : idx + 1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    x = torch.cat([torch.cat(_, 0) for _ in datas]).to(device)
    y = torch.cat([torch.cat(_) for _ in labels]).view(-1).to(device)
    return x, y


def get_layer_metric_array(net, metric, mode):
    metric_array = []

    for layer in net.modules():
        if mode == "channel" and hasattr(layer, "dont_ch_prune"):
            continue
        #if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        if isinstance(layer, (nn.Conv1d, nn.LSTM, nn.GRU, nn.Linear, )):
            metric_value = metric(layer)
            if isinstance(metric_value, list):
                metric_array.extend(metric(layer))
            else:
                metric_array.append(metric(layer))

    return metric_array


def reshape_elements(elements, shapes, device):
    def broadcast_val(elements, shapes):
        ret_grads = []
        for e, sh in zip(elements, shapes):
            ret_grads.append(
                torch.stack([torch.Tensor(sh).fill_(v) for v in e], dim=0).to(device)
            )
        return ret_grads

    if type(elements[0]) == list:
        outer = []
        for e, sh in zip(elements, shapes):
            outer.append(broadcast_val(e, sh))
        return outer
    else:
        return broadcast_val(elements, shapes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
