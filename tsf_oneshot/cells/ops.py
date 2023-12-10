from collections.abc import Iterable
from typing import Optional, Union
import torch
from torch import nn

from tsf_oneshot.cells.encoders import PRIMITIVES_Encoder
from tsf_oneshot.cells.decoders import PRIMITIVES_Decoder


class MixedEncoderOp(nn.Module):
    """ Mixed operation """
    available_ops = PRIMITIVES_Encoder

    def __init__(self, d_model: int,
                 PRIMITIVES: list[str],
                 OPS_kwargs: dict[str, dict] | None = None):
        super().__init__()

        self._ops = nn.ModuleList()

        for primitive in PRIMITIVES:
            op_kwargs = {"d_model": d_model}
            if OPS_kwargs is not None and primitive in OPS_kwargs:
                op_kwargs.update(OPS_kwargs[primitive])
            op = self.available_ops[primitive](**op_kwargs)

            self._ops.append(op)

        self._ops_names = PRIMITIVES

    def forward(self, weights, alpha_prune_threshold=0.0, **model_input_kwargs):
        """
        Args:
            weights: weight for each operation. We note that for subnetworks, "weights" indicate the architecture
                weights of the un-split network, thereby, we need
            alpha_prune_threshold: prune ops during forward pass if alpha below threshold
        """
        outputs = []
        for weight, op in zip(weights, self._ops):
            if weight > alpha_prune_threshold:
                outputs.append(list(weight * out for out in op(**model_input_kwargs)))
        return list(sum(out) for out in zip(*outputs))

    @property
    def op_names(self):
        return self._ops_names

    def __iter__(self) -> Iterable[nn.Module]:
        return iter([self._ops[idx] for idx in range(len(self._ops_idx))])

    def __getitem__(self, idx: Union[int, slice]) -> nn.Module:
        return self._ops[idx]

    def __len__(self):
        if self._ops_idx is None:
            return len(self._ops)
        else:
            return len(self._ops_idx)


class MixedDecoderOps(MixedEncoderOp):
    available_ops = PRIMITIVES_Decoder

    def forward(self, weights, alpha_prune_threshold=0.0, **model_input_kwargs):
        """
        Args:
            weights: weight for each operation. We note that for subnetworks, "weights" indicate the architecture
                weights of the un-split network, thereby, we need
            alpha_prune_threshold: prune ops during forward pass if alpha below threshold
        """
        return sum(
            w * op(**model_input_kwargs) for w, op in zip(weights, self._ops) if w > alpha_prune_threshold
        )