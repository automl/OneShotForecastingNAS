import copy
import os
import json
from pathlib import Path
import inspect

import torch
from torch import nn
from tsf_oneshot.networks.components import AbstractSearchEncoder, LinearDecoder, series_decomp
from tsf_oneshot.networks.combined_net_utils import (
    get_kwargs,
    forward_concat_net,
    forward_parallel_net
)
from tsf_oneshot.cells.cells import SampledEncoderCell, SampledDecoderCell, SampledFlatEncoderCell
from tsf_oneshot.cells.ops import PRIMITIVES_Encoder
from tsf_oneshot.prediction_heads import PREDICTION_HEADs, FLATPREDICTION_HEADs


class SampledEncoder(AbstractSearchEncoder):
    def __init__(self,
                 d_input: int,
                 d_model: int,
                 operations: list[int],
                 has_edges: list[bool],
                 n_cells: int,
                 window_size: int,
                 forecasting_horizon:int,
                 n_nodes: int = 4,
                 n_cell_input_nodes: int = 1,
                 PRIMITIVES=PRIMITIVES_Encoder.keys(),
                 OPS_kwargs: dict[str, dict] = {},
                 ):
        nn.Module.__init__(self)
        self.d_input = d_input
        self.d_model = d_model
        self.n_cells = n_cells
        self.n_nodes = n_nodes

        self.operations = operations
        self.has_edges = has_edges
        self.ops = PRIMITIVES
        self.OPS_kwargs = OPS_kwargs
        self.n_cell_input_nodes = n_cell_input_nodes

        # self.embedding_layer = nn.Linear(d_input, d_model, bias=True)

        cells = []
        num_edges = None
        edge2index = None
        d_inputs = [d_input] * n_cell_input_nodes
        for i in range(n_cells):
            cell = self.get_cell(n_nodes=n_nodes,
                                 n_input_nodes=n_cell_input_nodes,
                                 d_model=d_model,
                                 d_inputs=d_inputs,
                                 window_size=window_size,
                                 forecasting_horizon=forecasting_horizon,
                                 operations=operations,
                                 has_edges=has_edges,
                                 PRIMITIVES=PRIMITIVES,
                                 cell_idx=i,
                                 OPS_kwargs=OPS_kwargs,
                                 )
            cells.append(cell)
            d_inputs = [*d_inputs[1:], cell.d_model]
            if num_edges is None:
                num_edges = cell.num_edges
                edge2index = cell.edge2index
        self.cells = nn.ModuleList(cells)

        self.num_edges = num_edges

        self._device = torch.device('cpu')

    def get_cell(self, **kwargs):
        return SampledEncoderCell(**kwargs)

    def save(self, base_path: Path):
        meta_info = {
            'input_dim': self.input_dims,
            'd_model': self.d_model,
            'operations': self.operations,
            'has_edges': self.has_edges,
            'n_cells': self.n_cells,
            'n_cell_input_nodes': self.n_cell_input_nodes,
            'PRIMITIVES': self.ops,
            'OPS_kwargs': self.OPS_kwargs,
        }

        if not base_path.exists():
            os.makedirs(base_path)

        with open(base_path / f'meta_info.json', 'w') as f:
            json.dump(meta_info, f)
        torch.save(self.state_dict(), base_path / f'model_weights.pth')

    def cells_forward(self, states: list[torch.Tensor], **kwargs):
        cell_out = None
        cell_intermediate_steps = []
        for cell in self.cells:
            cell_out = cell(s_previous=states, )
            states = [*states[1:], cell_out]
            cell_intermediate_steps.append(cell.intermediate_outputs)
        return cell_out, cell_intermediate_steps


class SampledDecoder(SampledEncoder):
    def __init__(self, **kwargs):
        super(SampledDecoder, self).__init__(**kwargs)

    def get_cell(self, **kwargs):
        return SampledDecoderCell(**kwargs)

    def cells_forward(self, states: list[torch.Tensor],
                      cells_encoder_output: list[torch.Tensor], net_encoder_output: torch.Tensor, **kwargs):
        cell_out = None
        for cell_encoder_output, cell in zip(cells_encoder_output, self.cells):
            cell_out = cell(s_previous=states,
                            cell_encoder_output=cell_encoder_output, net_encoder_output=net_encoder_output)
            states = [*states[1:], cell_out]
        return cell_out


class SampledNet(nn.Module):
    def __init__(self,
                 d_input_past: int,
                 d_input_future: int,
                 window_size:int,
                 forecasting_horizon: int,
                 d_model: int,
                 d_output: int,
                 n_cells: int,
                 n_nodes: int,
                 n_cell_input_nodes: int,
                 operations_encoder: list[int],
                 has_edges_encoder: list[bool],
                 operations_decoder: list[int],
                 has_edges_decoder: list[bool],
                 PRIMITIVES_encoder: list[str],
                 PRIMITIVES_decoder: list[str],
                 OPS_kwargs: dict[str, dict],
                 HEAD: str,
                 HEADs_kwargs: dict[str, dict],
                 DECODER: str = 'seq',
                 backcast_loss_ration: float = 0.0
                 ):
        super(SampledNet, self).__init__()
        self.meta_info = dict(
            d_input_past=d_input_past, d_input_future=d_input_future,
            d_model=d_model,
            OPS_kwargs=OPS_kwargs,
            d_output=d_output,
            n_cells=n_cells, n_nodes=n_nodes, n_cell_input_nodes=n_cell_input_nodes,
            operations_encoder=operations_encoder, has_edges_encoder=has_edges_encoder,
            operations_decoder=operations_decoder,
            has_edges_decoder=has_edges_decoder,
            PRIMITIVES_encoder=PRIMITIVES_encoder,
            PRIMITIVES_decoder=PRIMITIVES_decoder,
            HEAD=HEAD, HEADs_kwargs=HEADs_kwargs,
            DECODER=DECODER,
            backcast_loss_ration=backcast_loss_ration,
        )
        self.d_input_past = d_input_past
        self.d_input_future = d_input_future
        self.d_output = d_output
        self.d_model = d_model

        self.n_cells = n_cells
        self.n_nodes = n_nodes
        self.n_cell_input_nodes = n_cell_input_nodes

        self.PRIMITIVES_encoder = PRIMITIVES_encoder
        self.PRIMITIVES_decoder = PRIMITIVES_decoder
        self.operations_encoder = operations_encoder
        self.has_edges_encoder = has_edges_encoder
        self.operations_decoder = operations_decoder
        self.has_edges_decoder = has_edges_decoder

        self.HEAD = HEAD
        self.HEADs_kwargs = HEADs_kwargs

        self.DECODER = DECODER

        forecast_only = backcast_loss_ration == 0.0
        self.forecast_only = forecast_only
        self.backcast_loss_ration = backcast_loss_ration

        encoder_kwargs = dict(d_input=d_input_past,
                              d_model=d_model,
                              n_cells=n_cells,
                              n_nodes=n_nodes,
                              window_size=window_size,
                              forecasting_horizon=forecasting_horizon,
                              n_cell_input_nodes=n_cell_input_nodes,
                              PRIMITIVES=PRIMITIVES_encoder,
                              operations=operations_encoder,
                              has_edges=has_edges_encoder,
                              OPS_kwargs=OPS_kwargs,
                              )

        self.encoder: SampledEncoder = SampledEncoder(
            **encoder_kwargs
        )

        decoder_kwargs = dict(d_input=d_input_future,
                              d_model=d_model,
                              n_cells=n_cells,
                              n_nodes=n_nodes,
                              window_size=window_size,
                              forecasting_horizon=forecasting_horizon,
                              n_cell_input_nodes=n_cell_input_nodes,
                              PRIMITIVES=PRIMITIVES_decoder,
                              operations=operations_decoder,
                              has_edges=has_edges_decoder,
                              OPS_kwargs=OPS_kwargs,
                              )
        if DECODER == 'seq':
            self.decoder: SampledDecoder = SampledDecoder(
                **decoder_kwargs
            )
        else:
            linear_decoder_kwargs = {}
            if 'general' in OPS_kwargs:
                linear_decoder_kwargs.update(OPS_kwargs['general'])
            if 'linear_decoder' in OPS_kwargs:
                linear_decoder_kwargs.update(OPS_kwargs['linear_decoder'])
            self.decoder = LinearDecoder(window_size, forecasting_horizon,
                                         d_input_future=d_input_future, d_model=d_model, **linear_decoder_kwargs)

        self.head = PREDICTION_HEADs[HEAD](d_model=d_model, d_output=d_output, **HEADs_kwargs)

    def forward(self, x_past: torch.Tensor,
                x_future: torch.Tensor):
        net_encoder_output, cell_intermediate_steps = self.encoder(x_past)

        cell_decoder_out = self.decoder(x=x_future,
                                        cells_encoder_output=cell_intermediate_steps,
                                        net_encoder_output=net_encoder_output)

        forecast = self.head(cell_decoder_out)

        if self.forecast_only:
            return forecast

        backcast = self.head(net_encoder_output)

        return backcast, forecast

    def get_inference_prediction(self, prediction):
        return self.head.get_inference_pred(prediction)

    def get_training_loss(self, target, prediction):
        return self.head.loss(target, prediction)

    @torch.no_grad()
    def grad_norm(self):
        total_norm = 0.0
        for name, par in self.named_parameters():
            if par.grad is not None:
                param_norm = par.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def get_weight_optimizer_parameters(self, weight_decay: float):
        """
        We don not want the weights for layer norm to be appleid with weight deacay
        codes mainly from https://github.com/wpeebles/G.pt/blob/main/Gpt/models/transformer.py
        :param cfg_w_optimizer:
        :return:
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (
            torch.nn.LayerNorm,
        )

        all_sets = set()

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)
                all_sets.add(fpn)

        # decay.add('decoder._fsdp_wrapped_module.flat_param')
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay

        decay = decay - inter_params
        union_params = decay | no_decay
        # assert len(inter_params) == 0, \
        #    "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params),)
        # TODO rewrite this function!!
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict],
             "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict], "weight_decay": 0.0},
        ]

        return optim_groups

    def save(self, base_path: Path):

        if not base_path.exists():
            os.makedirs(base_path)

        with open(base_path / f'meta_info.json', 'w') as f:
            json.dump(self.meta_info, f)
        torch.save(self.state_dict(), base_path / f'model_weights.pth')

    @staticmethod
    def load(base_path: Path, device=torch.device('cpu'), model="SampledNet"):
        if model is None:
            with open(base_path / f'meta_info.json', 'r') as f:
                meta_info = json.load(f)
            model = SampledNet(**meta_info)
        if (base_path / f'model_weights.pth').exists():
            model.load_state_dict(torch.load(base_path / f'model_weights.pth', map_location=device))
        return model


class SampledFlatNet(SampledNet):
    def __init__(self,
                 window_size: int,
                 forecasting_horizon: int,
                 d_output: int,
                 n_cells: int,
                 n_nodes: int,
                 n_cell_input_nodes: int,
                 operations_encoder: list[int],
                 has_edges_encoder: list[bool],
                 PRIMITIVES_encoder: list[str],
                 OPS_kwargs: dict[str, dict],
                 HEAD: str,
                 HEADs_kwargs: dict[str, dict],
                 backcast_loss_ration: float = 0.0
                 ):
        nn.Module.__init__(self)
        self.meta_info = dict(
            window_size=window_size, forecasting_horizon=forecasting_horizon,
            OPS_kwargs=OPS_kwargs,
            n_cells=n_cells, n_nodes=n_nodes, n_cell_input_nodes=n_cell_input_nodes,
            operations_encoder=operations_encoder, has_edges_encoder=has_edges_encoder,
            PRIMITIVES_encoder=PRIMITIVES_encoder,
            HEAD=HEAD, HEADs_kwargs=HEADs_kwargs,
            backcast_loss_ration=backcast_loss_ration
        )
        self.d_output = d_output
        self.forecasting_horizon = forecasting_horizon
        self.window_size = window_size

        self.n_cells = n_cells
        self.n_nodes = n_nodes
        self.n_cell_input_nodes = n_cell_input_nodes

        self.PRIMITIVES_encoder = PRIMITIVES_encoder
        self.operations_encoder = operations_encoder
        self.has_edges_encoder = has_edges_encoder
        forecast_only = backcast_loss_ration == 0.0
        self.forecast_only = forecast_only
        self.backcast_loss_ration = backcast_loss_ration

        cells = []
        num_edges = None
        edge2index = None
        for i in range(n_cells):
            cell = self.get_cell(n_nodes=n_nodes,
                                 n_input_nodes=n_cell_input_nodes,
                                 window_size=window_size,
                                 forecasting_horizon=forecasting_horizon,
                                 operations=operations_encoder,
                                 has_edges=has_edges_encoder,
                                 PRIMITIVES=PRIMITIVES_encoder,
                                 is_last_cell=(i == n_cells - 1),
                                 OPS_kwargs=OPS_kwargs,
                                 )
            cells.append(cell)
            if num_edges is None:
                num_edges = cell.num_edges
        self.cells = nn.ModuleList(cells)

        self.num_edges = num_edges

        self.head = FLATPREDICTION_HEADs[HEAD](window_size=window_size, forecasting_horizon=forecasting_horizon,
                                               **HEADs_kwargs)

        self._device = torch.device('cpu')

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

    @staticmethod
    def get_cell(**kwargs):
        return SampledFlatEncoderCell(**kwargs)

    def forward(self, x_past: torch.Tensor,
                x_future: torch.Tensor,
                forward_only_with_net: bool = False):
        # we always transform the input multiple_series map into independent single series input
        batch_size = x_past.shape[0]

        # we only take the past targets
        x_past_ = x_past[:, :, :self.d_output]

        seasonal_init, trend_init = self.decompsition(x_past_)

        past_targets = torch.transpose(x_past_, -1, -2)
        future_targets = torch.zeros([*past_targets.shape[:-1], self.forecasting_horizon], device=past_targets.device,
                                     dtype=past_targets.dtype)
        embedding = torch.cat([past_targets, future_targets], dim=-1)
        states = [embedding]
        #"""
        for cell in self.cells:
            cell_out = cell(s_previous=states, )
            states = [*states[1:], cell_out]

        cell_out = cell_out.transpose(-1, -2)
        back_cast, fore_cast = torch.split(cell_out, [self.window_size, self.forecasting_horizon], dim=1)
        if forward_only_with_net:
            # TODO check if back_cast is required !
            if self.forecast_only:
                return fore_cast
            return back_cast, fore_cast
        else:
            if self.forecast_only:
                return self.head(fore_cast)
            return self.head(back_cast), self.head(fore_cast), torch.cat([trend_init, x_past[:, :, self.d_output:]], -1),

    @staticmethod
    def load(base_path: Path, device=torch.device('cpu'), model="SampledNet"):
        if model is None:
            with open(base_path / f'meta_info.json', 'r') as f:
                meta_info = json.load(f)
            model = SampledFlatEncoderCell(**meta_info)
        if (base_path / f'model_weights.pth').exists():
            model.load_state_dict(torch.load(base_path / f'model_weights.pth', map_location=device))
        return model


class AbstractMixedSampledNet(SampledNet):

    def __init__(self,
                 d_input_past: int,
                 d_input_future: int,
                 d_output: int,

                 d_model: int,
                 n_cells_seq: int,
                 n_nodes_seq: int,
                 n_cell_input_nodes_seq: int,
                 operations_encoder_seq: list[int],
                 has_edges_encoder_seq: list[bool],
                 operations_decoder_seq: list[int],
                 has_edges_decoder_seq: list[bool],
                 PRIMITIVES_encoder_seq: list[str],
                 PRIMITIVES_decoder_seq: list[str],
                 OPS_kwargs_seq: dict[str, dict],
                 DECODER_seq: str,

                 HEAD: str,
                 HEADs_kwargs_seq: dict[str, dict],

                 window_size: int,
                 forecasting_horizon: int,
                 n_cells_flat: int,
                 n_nodes_flat: int,
                 n_cell_input_nodes_flat: int,
                 operations_encoder_flat: list[int],
                 has_edges_encoder_flat: list[bool],
                 PRIMITIVES_encoder_flat: list[str],
                 OPS_kwargs_flat: dict[str, dict],
                 HEADs_kwargs_flat: dict[str, dict],

                 nets_weights: list[float],

                 backcast_loss_ration_seq: float = 0.0,
                 backcast_loss_ration_flat: float = 0.0,

                 ):
        all_kwargs = get_kwargs()

        nn.Module.__init__(self)

        self.meta_info = copy.copy(all_kwargs)

        all_kwargs = self.validate_input_kwargs(all_kwargs)

        # get arguments for the seq net
        seq_net_kwargs = {}
        for arg_name in inspect.signature(SampledNet.__init__).parameters:
            if arg_name != 'self':
                if arg_name in all_kwargs:
                    seq_net_kwargs[arg_name] = all_kwargs[arg_name]
                elif  f'{arg_name}_seq' in all_kwargs:
                    seq_net_kwargs[arg_name] = all_kwargs[f'{arg_name}_seq']

        self.seq_net = SampledNet( **seq_net_kwargs)

        # get arguments for the flat net
        flat_net_kwargs = {}
        for arg_name in inspect.signature(SampledFlatNet.__init__).parameters:
            if arg_name != 'self':
                if arg_name in all_kwargs:
                    flat_net_kwargs[arg_name] = all_kwargs[arg_name]
                else:
                    flat_net_kwargs[arg_name] = all_kwargs[f'{arg_name}_flat']

        self.flat_net = SampledFlatNet(**flat_net_kwargs)

        # this function helps us to directly
        self.d_output = d_output

        # TODO check how to properly handle the heads
        self.head = self.seq_net.head

        forecast_only_seq = backcast_loss_ration_seq == 0
        forecast_only_flat = backcast_loss_ration_flat == 0
        self.forecast_only = forecast_only_seq or forecast_only_flat
        self.backcast_loss_ration = max(backcast_loss_ration_seq, backcast_loss_ration_flat)

        self.nets_weights = nn.Parameter(torch.Tensor(nets_weights), requires_grad=False)

        self.decompose = series_decomp(kernel_size=25)

    def validate_input_kwargs(self, kwargs):
        return kwargs

    def transform_nets_weights(self):
        res = torch.nn.functional.sigmoid(self.nets_weights)
        return res / torch.sum(res)

    @staticmethod
    def load(base_path: Path, device=torch.device('cpu'), model="SampledNet"):
        raise NotImplementedError


class MixedParallelSampledNet(AbstractMixedSampledNet):
    @staticmethod
    def load(base_path: Path, device=torch.device('cpu'), model="SampledNet"):
        if model is None:
            with open(base_path / f'meta_info.json', 'r') as f:
                meta_info = json.load(f)
            model = MixedParallelSampledNet(**meta_info)
        if (base_path / f'model_weights.pth').exists():
            model.load_state_dict(torch.load(base_path / f'model_weights.pth', map_location=device))
        return model

    def forward(self, x_past: torch.Tensor,
                x_future: torch.Tensor):
        return forward_parallel_net(
            flat_net=self.flat_net,
            seq_net=self.seq_net,
            decompose=self.decompose,
            x_past=x_past,
            x_future=x_future,
            out_weights=self.transform_nets_weights()
        )


class MixedConcatSampledNet(AbstractMixedSampledNet):
    def __init__(self, **kwargs):
        super(MixedConcatSampledNet, self).__init__(**kwargs)
        self.flat_net.forecast_only = False # TODO check if there is a better way to handle this

    def validate_input_kwargs(self, kwargs):
        kwargs = super().validate_input_kwargs(kwargs)
        kwargs['forecast_only_flat'] = False
        return kwargs

    @staticmethod
    def load(base_path: Path, device=torch.device('cpu'), model="SampledNet"):
        if model is None:
            with open(base_path / f'meta_info.json', 'r') as f:
                meta_info = json.load(f)
            model = MixedConcatSampledNet(**meta_info)
        if (base_path / f'model_weights.pth').exists():
            model.load_state_dict(torch.load(base_path / f'model_weights.pth', map_location=device))
        return model

    def forward(self, x_past: torch.Tensor,
                x_future: torch.Tensor):
        return forward_concat_net(
            flat_net=self.flat_net,
            seq_net=self.seq_net,
            x_past=x_past,
            x_future=x_future,
            decompose=self.decompose,
            out_weights=self.transform_nets_weights()
        )
