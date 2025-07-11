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

from torch import nn

from . import measure
from ..p_utils import get_layer_metric_array


@measure("l2_norm", copy_net=False, mode="param")
def get_l2_norm_array(net,  inputs_past, inputs_future, targets, mode, split_data=1, **kwargs):
    def get_l2norm(layer):
        if isinstance(layer, (nn.GRU, nn.LSTM)):
            grad_l2_norm = []
            for weights in layer.all_weights:
                for weight in weights:
                    grad_l2_norm.append(weight.norm())
        else:
            grad_l2_norm = layer.weight.norm()
        return grad_l2_norm
    return get_layer_metric_array(net, get_l2norm, mode=mode)
