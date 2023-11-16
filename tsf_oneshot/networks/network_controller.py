import os
import json

from typing import Union
import torch
from torch import nn

from pathlib import Path
from tsf_oneshot.networks.networks import (
    ForecastingDARTSNetwork, ForecastingGDASNetwork,
)
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
                 PRIMITIVES: list[str],
                 HEADS: list[str],
                 HEADS_kwargs: dict[str, dict],
                 val_loss_criterion: str = 'mse'
                 ):
        self.meta_info = dict(
            d_input_past=d_input_past, d_input_future=d_input_future,
            d_model=d_model,
            d_output=d_output,
            n_cells=n_cells, n_nodes=n_nodes, n_cell_input_nodes=n_cell_input_nodes,
            PRIMITIVES=PRIMITIVES, HEADS=HEADS, HEADS_kwargs=HEADS_kwargs,
            val_loss_criterion=val_loss_criterion
        )
        super(AbstractForecastingNetworkController, self).__init__()
        self.net = self.net_type(d_input_past=d_input_past, d_input_future=d_input_future,
                                 d_model=d_model,
                                 d_output=d_output,
                                 n_cells=n_cells, n_nodes=n_nodes, n_cell_input_nodes=n_cell_input_nodes,
                                 PRIMITIVES=PRIMITIVES, HEADS=HEADS, HEADS_kwargs=HEADS_kwargs)

        self.arch_p_encoder = nn.Parameter(1e-3 * torch.randn(self.net.encoder_n_edges, len(PRIMITIVES)))
        self.arch_p_decoder = nn.Parameter(1e-3 * torch.randn(self.net.decoder_n_edges, len(PRIMITIVES)))
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

    @torch.no_grad()
    def gard_norm_weights(self):
        total_norm = 0.0
        for name, p in self.named_weights():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                if param_norm ** 2 > 10:
                    print(param_norm ** 2)
                    print(name)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    @torch.no_grad()
    def gard_norm_alphas(self):
        total_norm = 0.0
        for name, p in self.named_arch_parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
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
    def load(base_path: Path, device=torch.device('cpu')):
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
                 PRIMITIVES: list[str],
                 HEADS: list[str],
                 HEADS_kwargs: dict[str, dict] = {},
                 normalizer: dict = {},
                 val_loss_criterion: str = 'mse'
                 ):
        super(ForecastingDARTSNetworkController, self).__init__(d_input_past=d_input_past,
                                                                d_input_future=d_input_future,
                                                                d_model=d_model, d_output=d_output,
                                                                n_cells=n_cells, n_nodes=n_nodes,
                                                                n_cell_input_nodes=n_cell_input_nodes,
                                                                PRIMITIVES=PRIMITIVES, HEADS=HEADS,
                                                                HEADS_kwargs=HEADS_kwargs,
                                                                val_loss_criterion=val_loss_criterion)
        self.normalizer = get_normalizer(normalizer)

    def get_w_dag(self, arch_p: torch.Tensor):
        return apply_normalizer(self.normalizer, arch_p)

    def get_training_loss(self, targets: torch.Tensor, predictions: list[torch.Tensor], w_dag: torch.Tensor):
        return sum(
            w * head.loss(targets=targets, predictions=pred) for w, pred, head in zip(w_dag[0], predictions, self.net.heads)
        )

    def get_validation_loss(self, targets: torch.Tensor, predictions: list[torch.Tensor], w_dag: torch.Tensor):
        inference_prediction = sum(
            w * head.get_inference_pred(predictions=pred) for w, pred, head in zip(w_dag[0],
                                                                                   predictions,
                                                                                   self.net.heads)
        )
        return self.criterion(inference_prediction, targets)

    @staticmethod
    def load(base_path: Path, device=torch.device('cpu')):
        with open(base_path / f'meta_info.json', 'r') as f:
            meta_info = json.load(f)
        model = ForecastingDARTSNetworkController(**meta_info)
        model.load_state_dict(torch.load(base_path / f'model_weights.pth', map_location=device))
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
            hardwts[_ie] * head.loss(targets, predictions) if _ie == index else hardwts[_ie]
            for _ie, head in enumerate(self.heads)
        )

    def get_validation_loss(self,
                            targets: torch.Tensor,
                            predictions: Union[list[torch.Tensor], torch.Tensor, torch.distributions.Distribution],
                            w_dag: tuple[torch.Tensor, torch.Tensor]):
        hardwts, index = w_dag
        inference_prediction = sum(
            hardwts[_ie] * head.get_inference_pred(targets, predictions) if _ie == index else hardwts[_ie]
            for _ie, head in enumerate(self.heads)
        )
        return self.criterion(inference_prediction, targets)

    @staticmethod
    def load(base_path: Path, device=torch.device('cpu')):
        with open(base_path / f'meta_info.json', 'r') as f:
            meta_info = json.load(f)
        model = ForecastingGDASNetworkController(**meta_info)
        model.load_state_dict(torch.load(base_path / f'model_weights.pth', map_location=device))
        return model
