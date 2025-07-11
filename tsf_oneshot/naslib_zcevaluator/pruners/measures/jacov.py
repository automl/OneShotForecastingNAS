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
"""
This contains implementations of jacov based on
https://github.com/BayesWatch/nas-without-training (jacov).
This script was based on
https://github.com/SamsungLabs/zero-cost-nas/blob/main/foresight/pruners/measures/jacob_cov.py
We found this version of jacov tends to perform
better.
Author: Robin Ru @ University of Oxford
"""

import torch
import numpy as np

from . import measure


def get_batch_jacobian(net, x_past, x_future, target):
    net.zero_grad()

    x_past.requires_grad_(True)
    #x_future.requires_grad_(True)

    y = net(x_past, x_future)
    if isinstance(y, list):
        # In this case, it is a quantile loss, we just check the intermediate output
        y = y[1]

    y.backward(torch.ones_like(y))
    jacob = x_past.grad.detach()

    return jacob, target.detach()


def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1.0 / (v + k))


@measure("jacov", bn=True)
def compute_jacob_cov(net,  inputs_past, inputs_future, targets, split_data=1, loss_fn=None):
    try:
        # Compute gradients (but don't apply them)
        jacobs, labels = get_batch_jacobian(net, inputs_past, inputs_future, targets)
        jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
        jc = eval_score(jacobs, labels)
    except Exception as e:
        print(e)
        jc = np.nan

    return jc
