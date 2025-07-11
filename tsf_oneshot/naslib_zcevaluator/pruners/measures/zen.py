#Copyright (C) 2010-2021 Alibaba Group Holding Limited.
# =============================================================================

import torch
import numpy as np

from torch import nn
from . import measure


def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if m.weight is None:
                    continue
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GRU, nn.LSTM)):
                for weights in m.all_weights:
                    for weight in weights:
                        nn.init.normal_(weight)
                # if hasattr(m, 'bias') and m.bias is not None:
                #     m.bias = False
                nn.init.zeros_(m.bias_ih_l0)
                nn.init.zeros_(m.bias_hh_l0)
                if m.bidirectional:
                    nn.init.zeros_(m.bias_ih_l0_reverse)
                    nn.init.zeros_(m.bias_hh_l0_reverse)
            else:
                continue

    return net

@measure("zen", bn=True)
def compute_zen_score(net, inputs_past, inputs_future, targets, loss_fn=None, split_data=1,
                      repeat=1, mixup_gamma=1e-2, fp16=False):
    nas_score_list = []

    device = inputs_past.device
    dtype = torch.half if fp16 else torch.float32

    with torch.no_grad():
        for repeat_count in range(repeat):
            network_weight_gaussian_init(net)
            input_past = torch.randn(size=list(inputs_past.shape), device=device, dtype=dtype)
            input2_past = torch.randn(size=list(inputs_past.shape), device=device, dtype=dtype)
            mixup_input_past = input_past + mixup_gamma * input2_past

            input_future = torch.randn(size=list(inputs_future.shape), device=device, dtype=dtype)
            input2_future = torch.randn(size=list(inputs_future.shape), device=device, dtype=dtype)
            mixup_input_future = input_future + mixup_gamma * input2_future

            output = net(input_past, input_future)
            mixup_output = net(mixup_input_past, mixup_input_future)
            if isinstance(mixup_output, list):
                mixup_output = mixup_output[1]
            if isinstance(output, list):
                output = output[1]

            nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2])
            nas_score = torch.mean(nas_score)

            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for m in net.modules():
                if isinstance(m, nn.BatchNorm1d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)
                pass
            pass
            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            nas_score_list.append(float(nas_score))

    avg_nas_score = float(np.mean(nas_score_list))

    return avg_nas_score
