from collections.abc import Iterable
from typing import Union
from torch import nn

from tsf_oneshot.cells.encoders import (
    PRIMITIVES_Encoder,
    PRIMITIVES_FLAT_ENCODER,
    NBEATSModule,
    NBEATS_DEFAULT_THETA_DIMS
)
from tsf_oneshot.cells.decoders import PRIMITIVES_Decoder


class MixedEncoderOps(nn.Module):
    """ Mixed operation """
    available_ops = PRIMITIVES_Encoder

    def __init__(self, d_model: int,
                 PRIMITIVES: list[str],
                 OPS_kwargs: dict[str, dict] | None = None,
                 kwargs_general: dict | None = None,
                 ):
        super().__init__()

        self._ops = nn.ModuleList()

        for primitive in PRIMITIVES:
            op_kwargs = {"d_model": d_model}
            if kwargs_general is not None:
                op_kwargs.update(kwargs_general)
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


class MixedFlatEncoderOps(MixedEncoderOps):
    available_ops = PRIMITIVES_FLAT_ENCODER

    def __init__(self, window_size: int,
                 forecasting_horizon: int,
                 PRIMITIVES: list[str],
                 OPS_kwargs: dict[str, dict] | None = None,
                 kwargs_general: dict | None = None):
        nn.Module.__init__(self)

        self.window_size = window_size

        self._ops = nn.ModuleList()

        self.flat_op_ids = []
        self.nbeats_block_ids = []

        has_nbeats_modules: bool = False
        nbeats_model = None
        for i, primitive in enumerate(PRIMITIVES):
            op_kwargs = {"window_size": window_size,
                         "forecasting_horizon": forecasting_horizon}
            if kwargs_general is not None:
                op_kwargs.update(kwargs_general)
            if OPS_kwargs is not None and primitive in OPS_kwargs:
                op_kwargs.update(OPS_kwargs[primitive])
            if not primitive.startswith('nbeats'):
                op = self.available_ops[primitive](**op_kwargs)
                self._ops.append(op)
            else:
                nbeats_type = primitive.split('_')[-1]
                if 'thetas_dim' not in op_kwargs:
                    op_kwargs['thetas_dim'] = NBEATS_DEFAULT_THETA_DIMS[nbeats_type]
                op = self.available_ops[primitive](has_fc_layers=False,
                                                   **op_kwargs)
                if nbeats_model is None:
                    nbeats_model: NBEATSModule = op
                self._ops.append(op)
                has_nbeats_modules = True

        if has_nbeats_modules:
            # all the nbeats models share the same fc layers to reduce computation complexity
            self.nbeats_fc_layers = NBEATSModule.generate_fc_layers(feature_in=window_size,
                                                                    num_fc_layers=nbeats_model.num_fc_layers,
                                                                    width=nbeats_model.width,
                                                                    norm_type=nbeats_model.norm_type,
                                                                    dropout=nbeats_model.dropout
                                                                    )
        self.has_nbeats_modules = has_nbeats_modules

        self._ops_names = PRIMITIVES

    def forward(self, weights, alpha_prune_threshold=0.0, **model_input_kwargs):
        """
        Args:
            weights: weight for each operation. We note that for subnetworks, "weights" indicate the architecture
                weights of the un-split network, thereby, we need
            alpha_prune_threshold: prune ops during forward pass if alpha below threshold
        """
        if self.has_nbeats_modules:
            x_fc_output = self.nbeats_fc_layers(model_input_kwargs['x_past'][:, :, :self.window_size])
        else:
            x_fc_output = None
        return sum(
            weight * op(**model_input_kwargs, x_fc_output=x_fc_output) for weight, op in zip(weights, self._ops) if weight > alpha_prune_threshold
        )


class MixedDecoderOps(MixedEncoderOps):
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