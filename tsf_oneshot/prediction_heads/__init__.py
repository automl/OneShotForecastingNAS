from collections.abc import Iterable
from typing import Union

import torch
from torch import nn

from tsf_oneshot.prediction_heads.heads import (
    QuantileHead, MSEOutput, MAEOutput, GammaOutput, PoissonOutput, NormalOutput, StudentTOutput
)

PREDICTION_HEADS = {
    "quantile": QuantileHead,
    "mse": MSEOutput,
    "mae": MAEOutput,
    "normal": NormalOutput,
    "studentT": StudentTOutput,
}


class MixedHead(nn.Module):
    """ Mixed Forecasting Heads"""
    available_ops = PREDICTION_HEADS

    def __init__(self,
                 d_model: int,
                 d_output: int,
                 PRIMITIVES: list[str],
                 OPS_kwargs: dict[str, dict] | None = None):
        super().__init__()

        self._ops = nn.ModuleList()

        for primitive in PRIMITIVES:
            op_kwargs = {"d_model": d_model, 'd_output': d_output}
            if primitive in OPS_kwargs:
                op_kwargs.update(OPS_kwargs[primitive])
            op = self.available_ops[primitive](**op_kwargs)

            self._ops.append(op)

        self._ops_names = PRIMITIVES

    def forward(self, x, weights, alpha_prune_threshold=0.0):
        """
        Args:
            weights: weight for each operation. We note that for subnetworks, "weights" indicate the architecture
                weights of the un-split network, thereby, we need
            alpha_prune_threshold: prune ops during forward pass if alpha below threshold
        """
        return sum(
            w * op(x) for w, op in zip(weights, self._ops) if w > alpha_prune_threshold
        )

    @property
    def op_names(self):
        return self._ops_names

    def __getitem__(self, idx: Union[int, slice]) -> nn.Module:
        return self._ops[idx]

    def __len__(self):
        return len(self._ops)
