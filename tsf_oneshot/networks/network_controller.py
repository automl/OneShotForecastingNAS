import os
import json

from typing import Union, Optional
import torch
from torch import nn

from pathlib import Path
from tsf_oneshot.networks.networks import (
    ForecastingDARTSNetwork, ForecastingGDASNetwork,
)
from tsf_oneshot.networks.components import EmbeddingLayer
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
                 forecasting_horizon: int,
                 d_model: int,
                 d_output: int,
                 n_cells: int,
                 n_nodes: int,
                 n_cell_input_nodes: int,
                 PRIMITIVES_encoder: list[str],
                 PRIMITIVES_decoder: list[str],
                 HEADS: list[str],
                 HEADS_kwargs: dict[str, dict] = {},
                 val_loss_criterion: str = 'mse'
                 ):
        self.meta_info = dict(
            d_input_past=d_input_past, d_input_future=d_input_future,
            d_model=d_model,
            forecasting_horizon=forecasting_horizon,
            d_output=d_output,
            n_cells=n_cells, n_nodes=n_nodes, n_cell_input_nodes=n_cell_input_nodes,
            PRIMITIVES_encoder=PRIMITIVES_encoder,
            PRIMITIVES_decoder=PRIMITIVES_decoder,
            HEADS=HEADS, HEADS_kwargs=HEADS_kwargs,
            val_loss_criterion=val_loss_criterion
        )
        super(AbstractForecastingNetworkController, self).__init__()
        self.net = self.net_type(d_input_past=d_input_past, d_input_future=d_input_future,
                                 d_model=d_model,
                                 forecasting_horizon=forecasting_horizon,
                                 d_output=d_output,
                                 n_cells=n_cells, n_nodes=n_nodes, n_cell_input_nodes=n_cell_input_nodes,
                                 PRIMITIVES_encoder=PRIMITIVES_encoder,
                                 PRIMITIVES_decoder=PRIMITIVES_decoder,
                                 HEADS=HEADS, HEADS_kwargs=HEADS_kwargs)

        self.arch_p_encoder = nn.Parameter(1e-3 * torch.randn(self.net.encoder_n_edges, len(PRIMITIVES_encoder)))
        self.arch_p_decoder = nn.Parameter(1e-3 * torch.randn(self.net.decoder_n_edges, len(PRIMITIVES_decoder)))
        self.arch_p_heads = nn.Parameter(1e-3 * torch.randn(1, len(HEADS)))

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

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor, return_w_head: bool=False):
        w_dag_encoder = self.get_w_dag(self.arch_p_encoder)
        w_dag_decoder = self.get_w_dag(self.arch_p_decoder)
        w_dag_head = self.get_w_dag(self.arch_p_heads)

        prediction = self.net(x_past=x_past, x_future=x_future, arch_p_encoder=w_dag_encoder,
                              arch_p_decoder=w_dag_decoder, arch_p_heads=w_dag_head)

        if return_w_head:
            return prediction, w_dag_head
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

    def get_weight_optimizer(self, cfg_w_optimizer):
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
        #assert len(inter_params) == 0, \
        #    "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
                % (str(param_dict.keys() - union_params), )
        # TODO rewrite this function!!
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict], "weight_decay": cfg_w_optimizer.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict], "weight_decay": 0.0},
        ]
        if cfg_w_optimizer.type == 'adamw':
            w_optimizer = torch.optim.AdamW(
                params=optim_groups,
                lr=cfg_w_optimizer.lr,
                betas=(cfg_w_optimizer.beta1, cfg_w_optimizer.beta2),
            )
        elif cfg_w_optimizer.type == 'sgd':
            w_optimizer = torch.optim.SGD(
                params=optim_groups,
                lr=cfg_w_optimizer.lr,
                momentum=cfg_w_optimizer.momentum,
            )
        else:
            raise NotImplementedError
        
        return w_optimizer


    @torch.no_grad()
    def grad_norm_weights(self):
        total_norm = 0.0
        for name, par in self.named_weights():
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
    def load(base_path: Path, device=torch.device('cpu'), model: Optional["AbstractForecastingNetworkController"] = None):
        raise NotImplementedError


class ForecastingDARTSNetworkController(AbstractForecastingNetworkController):
    net_type = ForecastingDARTSNetwork

    def __init__(self,
                 d_input_past: int,
                 d_input_future: int,
                 forecasting_horizon: int,
                 d_model: int,
                 d_output: int,
                 n_cells: int,
                 n_nodes: int,
                 n_cell_input_nodes: int,
                 PRIMITIVES_encoder: list[str],
                 PRIMITIVES_decoder: list[str],
                 HEADS: list[str],
                 HEADS_kwargs: dict[str, dict] = {},
                 normalizer: dict = {},
                 val_loss_criterion: str = 'mse'
                 ):
        super(ForecastingDARTSNetworkController, self).__init__(d_input_past=d_input_past,
                                                                d_input_future=d_input_future,
                                                                forecasting_horizon=forecasting_horizon,
                                                                d_model=d_model, d_output=d_output,
                                                                n_cells=n_cells, n_nodes=n_nodes,
                                                                n_cell_input_nodes=n_cell_input_nodes,
                                                                PRIMITIVES_encoder=PRIMITIVES_encoder,
                                                                PRIMITIVES_decoder=PRIMITIVES_decoder,
                                                                HEADS=HEADS,
                                                                HEADS_kwargs=HEADS_kwargs,
                                                                val_loss_criterion=val_loss_criterion)
        self.normalizer = get_normalizer(normalizer)

    def get_w_dag(self, arch_p: torch.Tensor):
        return apply_normalizer(self.normalizer, arch_p)

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
        inference_prediction = [
            w * head.get_inference_pred(predictions=pred) for w, pred, head in zip(w_dag[0],
                                                                                   predictions,
                                                                                   self.net.heads)
        ]
        return self.criterion(sum(inference_prediction), targets)

    @staticmethod
    def load(base_path: Path, device=torch.device('cpu'), model: Optional["ForecastingDARTSNetworkController"] = None):
        if model is None:
            with open(base_path / f'meta_info.json', 'r') as f:
                meta_info = json.load(f)
            model = ForecastingDARTSNetworkController(**meta_info)
        model.load_state_dict( torch.load(base_path / f'model_weights.pth', map_location=device))
        return model


class ForecastingGDASNetworkController(AbstractForecastingNetworkController):
    tau: float = 10
    net_type = ForecastingGDASNetwork

    def get_w_dag(self, arch_p: torch.Tensor):
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
