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

def SoftMax(logits, params, dim=-1):

    # temperature annealing
    if params["temp_anneal_mode"] == "linear":
        # linear temperature annealing (Note: temperature -> zero results in softmax -> argmax)
        temperature = params["t_max"] - params["curr_step"] * (
            params["t_max"] - params["t_min"]
        ) / (params["max_steps"] - 1)
        assert temperature > 0
    else:
        temperature = 1.0
    return F.softmax(logits / temperature, dim=dim)


def ReLUSoftMax(logits, params, dim=-1):
    lamb = params["curr_step"] / (params["max_steps"] - 1)
    return (1.0 - lamb) * F.softmax(logits, dim=dim) + lamb * (
        F.relu(logits) / torch.sum(F.relu(logits), dim=dim, keepdim=True)
    )


def GumbelSoftMax(logits, params, dim=-1):
    # NOTE: dim argument does not exist for gumbel_softmax in pytorch 1.0.1

    # temperature annealing
    if params["temp_anneal_mode"] == "linear":
        # linear temperature annealing (Note: temperature -> zero results in softmax -> argmax)
        temperature = params["t_max"] - params["curr_step"] * (
            params["t_max"] - params["t_min"]
        ) / (params["max_steps"] - 1)
        assert temperature > 0
    else:
        temperature = 1.0
    return F.gumbel_softmax(logits, temperature)


def get_normalizer(normalizer: dict | None):
    if normalizer is None:
        normalizer = {}
    if "name" not in normalizer.keys():
        normalizer["func"] = SoftMax
        normalizer["params"] = dict()
        normalizer["params"]["temp_anneal_mode"] = None
    elif normalizer["name"] == "softmax":
        normalizer["func"] = SoftMax
    elif normalizer["name"] == "relusoftmax":
        normalizer["func"] = ReLUSoftMax
    elif normalizer["name"] == "gumbel_softmax":
        normalizer["func"] = GumbelSoftMax
    else:
        raise RuntimeError(f"Unknown normalizer {normalizer['name']}")
    return normalizer


def apply_normalizer(normalizer: dict, alpha):
    if alpha.shape[1] == 1:
        return torch.ones_like(alpha)
    return normalizer['func'](alpha, normalizer["params"])


# for gdas
def gumble_sample(arch_parameters: torch.Tensor, tau: float):
    while True:
        gumbels = -torch.empty_like(arch_parameters).exponential_().log()
        logits = (arch_parameters.log_softmax(dim=-1) + gumbels) / tau
        probs = nn.functional.softmax(logits, dim=-1)
        if not (
                (torch.isinf(gumbels).any())
                or (torch.isinf(probs).any())
                or (torch.isnan(probs).any())
        ):
            index = probs.max(-1, keepdim=True)[1]
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            hardwts = one_h - probs.detach() + probs
            return hardwts, index
