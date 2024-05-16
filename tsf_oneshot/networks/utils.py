from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from tsf_oneshot.prediction_heads import MixedHead, MixedFlatHEADAS


def get_head_out(decoder_output: torch.Tensor, heads: MixedHead | MixedFlatHEADAS, head_idx: int| None = None):
    if head_idx is None:
        return list(head(decoder_output) for head in heads)
    else:
        return heads[head_idx](decoder_output)


# utils functions for DARTS and GDAS
# for gdas
# adapted from https://github.com/D-X-Y/AutoDL-Projects/blob/main/xautodl/models/cell_searchs/search_model_gdas.py
def gumble_sample(arch_parameters: torch.Tensor, tau: float):
    while True:
        gumbels = -torch.empty_like(arch_parameters).exponential_().log()
        if torch.isinf(gumbels).any():
            continue
        logits = (arch_parameters.log_softmax(dim=-1) + gumbels) / tau

        probs = nn.functional.softmax(logits, dim=-1)
        probs = torch.where(arch_parameters.isinf(), 0, probs)
        # to avoid the case when arch_parameters contains inf masks
        if not (
                (torch.isinf(probs).any())
                or (torch.isnan(probs).any())
        ):
            index = probs.max(-1, keepdim=True)[1]
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            hardwts = one_h - probs.detach() + probs
            return hardwts, index
