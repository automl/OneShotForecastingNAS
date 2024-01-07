import copy
import os
import json

from typing import Union, Optional
import torch
from torch import nn

from pathlib import Path
from tsf_oneshot.networks.networks import (
    ForecastingDARTSNetwork,
    ForecastingGDASNetwork,
    ForecastingDARTSFlatNetwork,
    ForecastingGDASFlatNetwork,
    ForecastingAbstractMixedNet,
    ForecastingDARTSMixedConcatNet,
    ForecastingGDASMixedConcatNet,
    ForecastingDARTSMixedParallelNet,
    ForecastingGDASMixedParallelNet
)
from tsf_oneshot.networks.combined_net_utils import get_kwargs
from tsf_oneshot.networks.utils import (
    apply_normalizer,
    get_normalizer,
    gumble_sample
)


class AbstractForecastingNetworkController(nn.Module):
    net_type = ForecastingDARTSNetwork

    def __init__(self,
                 d_input_past: int,
                 d_input_future: int,
                 d_model: int,
                 d_output: int,
                 n_cells: int,
                 n_nodes: int,
                 n_cell_input_nodes: int,
                 PRIMITIVES_encoder: list[str],
                 PRIMITIVES_decoder: list[str],
                 HEADs: list[str],
                 HEADs_kwargs: dict[str, dict] = {},
                 OPS_kwargs: dict[str, dict] = {},
                 val_loss_criterion: str = 'mse',
                 backcast_loss_ration: float = 0.0
                 ):
        self.meta_info = dict(
            d_input_past=d_input_past, d_input_future=d_input_future,
            d_model=d_model,
            OPS_kwargs=OPS_kwargs,
            d_output=d_output,
            n_cells=n_cells, n_nodes=n_nodes, n_cell_input_nodes=n_cell_input_nodes,
            PRIMITIVES_encoder=PRIMITIVES_encoder,
            PRIMITIVES_decoder=PRIMITIVES_decoder,
            HEADs=HEADs, HEADs_kwargs=HEADs_kwargs,
            val_loss_criterion=val_loss_criterion,
            backcast_loss_ration=backcast_loss_ration
        )
        forecast_only = backcast_loss_ration == 0
        self.forecast_only = forecast_only
        self.backcast_loss_ration = backcast_loss_ration

        super(AbstractForecastingNetworkController, self).__init__()
        self.net = self.net_type(d_input_past=d_input_past, d_input_future=d_input_future,
                                 d_model=d_model,
                                 OPS_kwargs=OPS_kwargs,
                                 d_output=d_output,
                                 n_cells=n_cells, n_nodes=n_nodes, n_cell_input_nodes=n_cell_input_nodes,
                                 PRIMITIVES_encoder=PRIMITIVES_encoder,
                                 PRIMITIVES_decoder=PRIMITIVES_decoder,
                                 HEADs=HEADs, HEADs_kwargs=HEADs_kwargs,
                                 forecast_only=forecast_only)

        self.arch_p_encoder = nn.Parameter(1e-3 * torch.randn(self.net.encoder_n_edges, len(PRIMITIVES_encoder)))
        self.arch_p_decoder = nn.Parameter(1e-3 * torch.randn(self.net.decoder_n_edges, len(PRIMITIVES_decoder)))
        self.arch_p_heads = nn.Parameter(1e-3 * torch.randn(1, len(HEADs)))

        # setup alphas list
        self._arch_ps = []
        for n, p in self.named_parameters():
            if "arch_p" in n:
                self._arch_ps.append((n, p))

        self.val_loss_criterion = val_loss_criterion
        if val_loss_criterion == 'mse':
            self.criterion = nn.MSELoss()
        elif val_loss_criterion == 'mae':
            self.criterion = nn.L1Loss()
        else:
            raise NotImplementedError

        # These masks are used to remove the corresponding operations during pertubation
        self.len_all_arch_p = [len(self.arch_p_encoder), len(self.arch_p_decoder), len(self.arch_p_heads)]

        self.mask_encoder = nn.Parameter(torch.zeros_like(self.arch_p_encoder), requires_grad=False)
        self.mask_decoder = nn.Parameter(torch.zeros_like(self.arch_p_decoder), requires_grad=False)
        self.mask_head = nn.Parameter(torch.zeros_like(self.arch_p_heads), requires_grad=False)

        self.all_masks = ['mask_encoder' 'mask_decoder', 'mask_head']
        self.len_all_arch_p = [len(getattr(self, mask)) for mask in self.all_masks]

        self.candidate_flags = [True] * sum(self.len_all_arch_p)

    def get_all_wags(self):
        w_dag_encoder = self.get_w_dag(self.arch_p_encoder + self.mask_encoder)
        w_dag_decoder = self.get_w_dag(self.arch_p_decoder + self.mask_decoder)
        w_dag_head = self.get_w_dag(self.arch_p_heads + self.mask_head)
        return dict(
            arch_p_encoder=w_dag_encoder,
            arch_p_decoder=w_dag_decoder,
            arch_p_heads=w_dag_head
        )

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor, return_w_head: bool = False):
        all_wags = self.get_all_wags()
        prediction = self.net(x_past=x_past, x_future=x_future, **all_wags)

        if return_w_head:
            return prediction, all_wags['arch_p_heads']
        return prediction

    def get_w_dag(self, arch_p: torch.Tensor):
        raise NotImplementedError

    def get_training_loss(self, targets: torch.Tensor, predictions: list[torch.Tensor], w_dag: torch.Tensor):
        raise NotImplementedError

    def get_validation_loss(self, targets: torch.Tensor, predictions: list[torch.Tensor], w_dag: torch.Tensor):
        raise NotImplementedError

    def get_individual_training_loss(self, targets: torch.Tensor, predictions: list[torch.Tensor]):
        raise NotImplementedError

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def named_weights_with_net(self):
        return self.named_parameters()

    def arch_parameters(self):
        for n, p in self._arch_ps:
            yield p

    def named_arch_parameters(self):
        for n, p in self._arch_ps:
            yield n, p

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

        for mn, m in self.net.named_modules():
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
        param_dict = {pn: p for pn, p in self.net.named_parameters()}
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

    @torch.no_grad()
    def grad_norm_weights(self):
        total_norm = 0.0
        for name, par in self.named_parameters():
            if par.grad is not None:
                param_norm = par.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    @torch.no_grad()
    def grad_norm_alphas(self):
        total_norm = 0.0
        for name, par in self.named_arch_parameters():
            if par.grad is not None:
                param_norm = par.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def save(self, base_path: Path):
        meta_info = self.meta_info

        if not base_path.exists():
            os.makedirs(base_path)

        with open(base_path / f'meta_info.json', 'w') as f:
            json.dump(meta_info, f)
        torch.save(self.state_dict(), base_path / f'model_weights.pth')

    @staticmethod
    def load(base_path: Path, device=torch.device('cpu'),
             model: Optional["AbstractForecastingNetworkController"] = None):
        raise NotImplementedError


class ForecastingDARTSNetworkController(AbstractForecastingNetworkController):
    net_type = ForecastingDARTSNetwork

    def __init__(self,
                 d_input_past: int,
                 d_input_future: int,
                 d_model: int,
                 d_output: int,
                 n_cells: int,
                 n_nodes: int,
                 n_cell_input_nodes: int,
                 PRIMITIVES_encoder: list[str],
                 PRIMITIVES_decoder: list[str],
                 HEADs: list[str],
                 HEADs_kwargs: dict[str, dict] = {},
                 normalizer: dict = {},
                 OPS_kwargs: dict[str, dict] = {},
                 val_loss_criterion: str = 'mse',
                 backcast_loss_ration: float = 0.0
                 ):
        super(ForecastingDARTSNetworkController, self).__init__(d_input_past=d_input_past,
                                                                d_input_future=d_input_future,
                                                                OPS_kwargs=OPS_kwargs,
                                                                d_model=d_model, d_output=d_output,
                                                                n_cells=n_cells, n_nodes=n_nodes,
                                                                n_cell_input_nodes=n_cell_input_nodes,
                                                                PRIMITIVES_encoder=PRIMITIVES_encoder,
                                                                PRIMITIVES_decoder=PRIMITIVES_decoder,
                                                                HEADs=HEADs,
                                                                HEADs_kwargs=HEADs_kwargs,
                                                                val_loss_criterion=val_loss_criterion,
                                                                backcast_loss_ration=backcast_loss_ration)
        self.normalizer = get_normalizer(normalizer)

    def get_w_dag(self, arch_p: torch.Tensor):
        w_dag = apply_normalizer(self.normalizer, arch_p)
        # this is applied when all arch_p is -torch.inf
        w_dag = torch.where(torch.isnan(w_dag), 0, w_dag)
        return w_dag

    def get_training_loss(self, targets: torch.Tensor, predictions: list[torch.Tensor], w_dag: torch.Tensor):
        losses = [w * head.loss(targets=targets, predictions=pred) for w, pred, head in
                  zip(w_dag[0], predictions, self.net.heads)]
        return sum(
            losses
        )

    def get_individual_training_loss(self, targets: torch.Tensor, predictions: list[torch.Tensor]):
        losses = [head.loss(targets=targets, predictions=pred) for pred, head in
                  zip(predictions, self.net.heads)]
        return losses

    def get_validation_loss(self, targets: torch.Tensor, predictions: list[torch.Tensor], w_dag: torch.Tensor):
        inference_prediction = self.get_inference_prediction(predictions, w_dag)
        return self.criterion(inference_prediction, targets)

    def get_inference_prediction(self, prediction, w_dag):
        inference_prediction = [
            w * head.get_inference_pred(predictions=pred) for w, pred, head in zip(w_dag[0],
                                                                                   prediction,
                                                                                   self.net.heads)
        ]
        return sum(inference_prediction)

    @staticmethod
    def load(base_path: Path, device=torch.device('cpu'), model: Optional["ForecastingDARTSNetworkController"] = None):
        if model is None:
            with open(base_path / f'meta_info.json', 'r') as f:
                meta_info = json.load(f)
            model = ForecastingDARTSNetworkController(**meta_info)
        model.load_state_dict(torch.load(base_path / f'model_weights.pth', map_location=device))
        return model


class AbstractForecastingFlatNetworkController(AbstractForecastingNetworkController):
    def __init__(self,
                 window_size: int,
                 forecasting_horizon: int,
                 n_cells: int,
                 n_nodes: int,
                 n_cell_input_nodes: int,
                 PRIMITIVES_encoder: list[str],
                 HEADs: list[str],
                 HEADs_kwargs,
                 OPS_kwargs: dict[str, dict] = {},
                 val_loss_criterion: str = 'mse',
                 backcast_loss_ration: float = 0.0,
                 ):
        self.meta_info = dict(
            window_size=window_size, forecasting_horizon=forecasting_horizon,
            OPS_kwargs=OPS_kwargs,
            n_cells=n_cells, n_nodes=n_nodes, n_cell_input_nodes=n_cell_input_nodes,
            PRIMITIVES_encoder=PRIMITIVES_encoder,
            HEADs=HEADs, HEADs_kwargs=HEADs_kwargs,
            val_loss_criterion=val_loss_criterion,
            backcast_loss_ration=backcast_loss_ration
        )
        super(AbstractForecastingNetworkController, self).__init__()
        forecast_only = backcast_loss_ration == 0

        self.net = self.net_type(window_size=window_size, forecasting_horizon=forecasting_horizon,
                                 OPS_kwargs=OPS_kwargs,
                                 n_cells=n_cells, n_nodes=n_nodes, n_cell_input_nodes=n_cell_input_nodes,
                                 PRIMITIVES_encoder=PRIMITIVES_encoder,
                                 HEADs=HEADs, HEADs_kwargs=HEADs_kwargs,
                                 forecast_only=forecast_only)

        self.arch_p_encoder = nn.Parameter(1e-3 * torch.randn(self.net.encoder_n_edges, len(PRIMITIVES_encoder)))
        self.arch_p_heads = nn.Parameter(1e-3 * torch.randn(1, len(HEADs)))

        # setup alphas list
        self._arch_ps = []
        for n, p in self.named_parameters():
            if "arch_p" in n:
                self._arch_ps.append((n, p))

        self.val_loss_criterion = val_loss_criterion
        if val_loss_criterion == 'mse':
            self.criterion = nn.MSELoss()
        elif val_loss_criterion == 'mae':
            self.criterion = nn.L1Loss()
        else:
            raise NotImplementedError

        self.mask_encoder = nn.Parameter(torch.zeros_like(self.arch_p_encoder), requires_grad=False)
        self.mask_head = nn.Parameter(torch.zeros_like(self.arch_p_heads), requires_grad=False)

        self.all_masks = ['mask_encoder', 'mask_head']
        self.len_all_arch_p = [len(getattr(self, mask)) for mask in self.all_masks]

        self.candidate_flags = [True] * sum(self.len_all_arch_p)

    def get_all_wags(self):
        w_dag_encoder = self.get_w_dag(self.arch_p_encoder)
        w_dag_head = self.get_w_dag(self.arch_p_heads)
        return dict(
            arch_p_encoder=w_dag_encoder,
            arch_p_heads=w_dag_head
        )


class ForecastingDARTSFlatNetworkController(AbstractForecastingFlatNetworkController,
                                            ForecastingDARTSNetworkController):
    net_type = ForecastingDARTSFlatNetwork

    def __init__(self,
                 window_size: int,
                 forecasting_horizon: int,
                 n_cells: int,
                 n_nodes: int,
                 n_cell_input_nodes: int,
                 PRIMITIVES_encoder: list[str],
                 HEADs: list[str],
                 HEADs_kwargs,
                 OPS_kwargs: dict[str, dict] = {},
                 normalizer={},
                 backcast_loss_ration: float = 0.0,
                 ):
        AbstractForecastingFlatNetworkController.__init__(self,
                                                          window_size=window_size,
                                                          forecasting_horizon=forecasting_horizon,
                                                          n_cells=n_cells, n_nodes=n_nodes,
                                                          n_cell_input_nodes=n_cell_input_nodes,
                                                          PRIMITIVES_encoder=PRIMITIVES_encoder,
                                                          HEADs=HEADs,
                                                          OPS_kwargs=OPS_kwargs,
                                                          HEADs_kwargs=HEADs_kwargs,
                                                          backcast_loss_ration=backcast_loss_ration)
        self.normalizer = get_normalizer(normalizer)

    @staticmethod
    def load(base_path: Path, device=torch.device('cpu'),
             model: Optional["ForecastingDARTSFlatNetworkController"] = None):
        if model is None:
            with open(base_path / f'meta_info.json', 'r') as f:
                meta_info = json.load(f)
            model = ForecastingDARTSFlatNetworkController(**meta_info)
        model.load_state_dict(torch.load(base_path / f'model_weights.pth', map_location=device))
        return model


class ForecastingGDASNetworkController(AbstractForecastingNetworkController):
    tau: float = 10
    net_type = ForecastingGDASNetwork

    def get_w_dag(self, arch_p: torch.Tensor):
        if arch_p.shape[1] == 0:
            return torch.ones_like(arch_p), torch.zeros_like(arch_p, dtype=torch.long)
        return gumble_sample(arch_p, self.tau)

    def get_training_loss(self,
                          targets: torch.Tensor,
                          predictions: Union[list[torch.Tensor], torch.Tensor, torch.distributions.Distribution],
                          w_dag: tuple[torch.Tensor, torch.Tensor]):
        hardwts, index = w_dag
        hardwts = hardwts[0]
        index = index[0]
        assert index is not None
        return sum(
            hardwts[_ie] * head.loss(targets, pred) if _ie == index else hardwts[_ie]
            for _ie, (head, pred) in enumerate(zip(self.heads, predictions))
        )

    def get_individual_training_loss(self, targets: torch.Tensor, predictions: list[torch.Tensor]):
        losses = [head.loss(targets=targets, predictions=pred) for pred, head in
                  zip(predictions, self.net.heads)]
        return losses

    def get_validation_loss(self,
                            targets: torch.Tensor,
                            predictions: Union[list[torch.Tensor], torch.Tensor, torch.distributions.Distribution],
                            w_dag: tuple[torch.Tensor, torch.Tensor]):
        hardwts, index = w_dag
        hardwts = hardwts[0]
        index = index[0]
        inference_prediction = sum(
            hardwts[_ie] * head.get_inference_pred(predictions=pred) if _ie == index else hardwts[_ie]
            for _ie, (head, pred) in enumerate(zip(self.net.heads, predictions))
        )
        return self.criterion(inference_prediction, targets)

    @staticmethod
    def load(base_path: Path, device=torch.device('cpu'), model: Optional["ForecastingGDASNetworkController"] = None):
        if model is None:
            with open(base_path / f'meta_info.json', 'r') as f:
                meta_info = json.load(f)
            model = ForecastingGDASNetworkController(**meta_info)
        model.load_state_dict(torch.load(base_path / f'model_weights.pth', map_location=device))
        return model


class ForecastingGDASFlatNetworkController(AbstractForecastingFlatNetworkController, ForecastingGDASNetworkController):
    net_type = ForecastingGDASFlatNetwork

    @staticmethod
    def load(base_path: Path, device=torch.device('cpu'),
             model: Optional["ForecastingGDASFlatNetworkController"] = None):
        if model is None:
            with open(base_path / f'meta_info.json', 'r') as f:
                meta_info = json.load(f)
            model = ForecastingGDASNetworkController(**meta_info)
        model.load_state_dict(torch.load(base_path / f'model_weights.pth', map_location=device))
        return model


class ForecastingAbstractMixedNetController(AbstractForecastingNetworkController):
    net_type = ForecastingAbstractMixedNet

    def __init__(self,
                 d_input_past: int,
                 d_input_future: int,
                 d_output: int,

                 d_model: int,
                 n_cells_seq: int,
                 n_nodes_seq: int,
                 n_cell_input_nodes_seq: int,
                 PRIMITIVES_encoder_seq: list[str],
                 PRIMITIVES_decoder_seq: list[str],
                 OPS_kwargs_seq: dict[str, dict],

                 window_size: int,
                 forecasting_horizon: int,
                 n_cells_flat: int,
                 n_nodes_flat: int,
                 n_cell_input_nodes_flat: int,
                 PRIMITIVES_encoder_flat: list[str],
                 OPS_kwargs_flat: dict[str, dict],

                 HEADs: list[str],
                 HEADs_kwargs_seq: dict[str, dict],
                 HEADs_kwargs_flat: dict[str, dict],

                 backcast_loss_ration_seq: float = 0.0,
                 backcast_loss_ration_flat: float = 0.0,
                 val_loss_criterion: str = 'mse'
                 ):
        all_kwargs = get_kwargs()
        self.meta_info = copy.copy(all_kwargs)

        self.validate_input_kwargs(all_kwargs)

        nn.Module.__init__(self)


        forecast_only_seq = backcast_loss_ration_seq == 0
        forecast_only_flat = backcast_loss_ration_flat == 0

        self.forecast_only = forecast_only_seq or forecast_only_flat

        self.backcast_loss_ration = max(backcast_loss_ration_seq, backcast_loss_ration_flat)

        self.net = self.net_type(**all_kwargs)
        self.heads = self.net.heads

        self.arch_p_encoder_seq = nn.Parameter(
            1e-3 * torch.randn(self.net.seq_net.encoder_n_edges, len(PRIMITIVES_encoder_seq))
        )
        self.arch_p_decoder_seq = nn.Parameter(
            1e-3 * torch.randn(self.net.seq_net.decoder_n_edges, len(PRIMITIVES_decoder_seq))
        )

        self.arch_p_encoder_flat = nn.Parameter(
            1e-3 * torch.randn(self.net.flat_net.encoder_n_edges, len(PRIMITIVES_encoder_flat))
        )
        self.arch_p_heads = nn.Parameter(1e-3 * torch.randn(1, len(HEADs)))

        # setup alphas list
        self._arch_ps = []
        for n, p in self.named_parameters():
            if "arch_p" in n:
                self._arch_ps.append((n, p))

        self.val_loss_criterion = val_loss_criterion
        if val_loss_criterion == 'mse':
            self.criterion = nn.MSELoss()
        elif val_loss_criterion == 'mae':
            self.criterion = nn.L1Loss()
        else:
            raise NotImplementedError

        # These masks are used to remove the corresponding operations during pertubation

        self.mask_encoder_seq = nn.Parameter(torch.zeros_like(self.arch_p_encoder_seq), requires_grad=False)
        self.mask_decoder_seq = nn.Parameter(torch.zeros_like(self.arch_p_decoder_seq), requires_grad=False)
        self.mask_encoder_flat = nn.Parameter(torch.zeros_like(self.arch_p_encoder_flat), requires_grad=False)
        self.mask_head = nn.Parameter(torch.zeros_like(self.arch_p_heads), requires_grad=False)

        self.all_masks = ['mask_encoder_seq', 'mask_decoder_seq', 'mask_encoder_flat', 'mask_head']
        self.len_all_arch_p = [len(getattr(self, mask)) for mask in self.all_masks]

        self.candidate_flags = [True] * sum(self.len_all_arch_p)

    def validate_input_kwargs(self, kwargs):
        backcast_loss_ration_seq = kwargs.pop('backcast_loss_ration_seq')
        backcast_loss_ration_flat = kwargs.pop('backcast_loss_ration_flat')
        # we need to ensure that the backcast loss ration of the two parts either consistent or
        assert backcast_loss_ration_flat == backcast_loss_ration_seq or \
               (backcast_loss_ration_flat * backcast_loss_ration_seq) == 0
        forecast_only_seq = backcast_loss_ration_seq == 0
        forecast_only_flat = backcast_loss_ration_flat == 0
        # update kwargs to fit the requirements of net_init function
        kwargs.pop('val_loss_criterion')
        kwargs['forecast_only_seq'] = forecast_only_seq
        kwargs['forecast_only_flat'] = forecast_only_flat
        return kwargs

    def get_all_wags(self):
        w_dag_encoder_seq = self.get_w_dag(self.arch_p_encoder_seq + self.mask_encoder_seq)
        w_dag_decoder_seq = self.get_w_dag(self.arch_p_decoder_seq + self.mask_decoder_seq)

        w_dag_encoder_flat = self.get_w_dag(self.arch_p_encoder_flat + self.mask_encoder_flat)

        w_dag_head = self.get_w_dag(self.arch_p_heads + self.mask_head)

        return dict(
            arch_p_encoder_seq=w_dag_encoder_seq,
            arch_p_decoder_seq=w_dag_decoder_seq,
            arch_p_heads_seq=w_dag_head,

            arch_p_encoder_flat=w_dag_encoder_flat,
            arch_p_heads_flat=w_dag_head
        )

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor, return_w_head: bool = False):
        all_wags = self.get_all_wags()
        prediction = self.net(x_past=x_past, x_future=x_future, **all_wags)

        if return_w_head:
            return prediction, all_wags['arch_p_heads_seq']
        return prediction


class ForecastingDARTSMixedParallelNetController(ForecastingAbstractMixedNetController,
                                                 ForecastingDARTSNetworkController):
    net_type = ForecastingDARTSMixedParallelNet

    def __init__(self,
                 d_input_past: int,
                 d_input_future: int,
                 d_output: int,

                 d_model: int,
                 n_cells_seq: int,
                 n_nodes_seq: int,
                 n_cell_input_nodes_seq: int,
                 PRIMITIVES_encoder_seq: list[str],
                 PRIMITIVES_decoder_seq: list[str],
                 OPS_kwargs_seq: dict[str, dict],

                 window_size: int,
                 forecasting_horizon: int,
                 n_cells_flat: int,
                 n_nodes_flat: int,
                 n_cell_input_nodes_flat: int,
                 PRIMITIVES_encoder_flat: list[str],
                 OPS_kwargs_flat: dict[str, dict],

                 HEADs: list[str],
                 HEADs_kwargs_seq: dict[str, dict],
                 HEADs_kwargs_flat: dict[str, dict],

                 backcast_loss_ration_seq: float = 0.0,
                 backcast_loss_ration_flat: float = 0.0,
                 val_loss_criterion: str = 'mse',

                 normalizer: dict = {}):
        all_kwargs = get_kwargs()
        all_kwargs.pop('normalizer')
        ForecastingAbstractMixedNetController.__init__(self, **all_kwargs)
        self.normalizer = get_normalizer(normalizer)

    @staticmethod
    def load(base_path: Path,
             device=torch.device('cpu'),
             model: Optional["ForecastingDARTSMixedParallelNetController"] = None):
        if model is None:
            with open(base_path / f'meta_info.json', 'r') as f:
                meta_info = json.load(f)
            model = ForecastingDARTSMixedParallelNetController(**meta_info)
        model.load_state_dict(torch.load(base_path / f'model_weights.pth', map_location=device))
        return model


class ForecastingDARTSMixedConcatNetController(ForecastingDARTSMixedParallelNetController,
                                               ForecastingDARTSNetworkController):
    net_type = ForecastingDARTSMixedConcatNet

    @staticmethod
    def load(base_path: Path,
             device=torch.device('cpu'),
             model: Optional["ForecastingDARTSMixedConcatNetController"] = None):
        if model is None:
            with open(base_path / f'meta_info.json', 'r') as f:
                meta_info = json.load(f)
            model = ForecastingDARTSMixedConcatNetController(**meta_info)
        model.load_state_dict(torch.load(base_path / f'model_weights.pth', map_location=device))
        return model


class ForecastingGDASMixedParallelNetController(ForecastingAbstractMixedNetController,
                                                ForecastingGDASNetworkController):
    net_type = ForecastingGDASMixedParallelNet

    @staticmethod
    def load(base_path: Path,
             device=torch.device('cpu'),
             model: Optional["ForecastingGDASMixedParallelNetController"] = None):
        if model is None:
            with open(base_path / f'meta_info.json', 'r') as f:
                meta_info = json.load(f)
            model = ForecastingGDASMixedParallelNetController(**meta_info)
        model.load_state_dict(torch.load(base_path / f'model_weights.pth', map_location=device))
        return model


class ForecastingGDASMixedConcatNetController(ForecastingAbstractMixedNetController,
                                              ForecastingGDASNetworkController):
    net_type = ForecastingGDASMixedConcatNet

    @staticmethod
    def load(base_path: Path,
             device=torch.device('cpu'),
             model: Optional["ForecastingGDASMixedConcatNetController"] = None):
        if model is None:
            with open(base_path / f'meta_info.json', 'r') as f:
                meta_info = json.load(f)
            model = ForecastingGDASMixedConcatNetController(**meta_info)
        model.load_state_dict(torch.load(base_path / f'model_weights.pth', map_location=device))
        return model
