import copy
from pathlib import Path
from typing import Type
import os
import json

import torch
from torch import nn

from tsf_oneshot.cells.cells import SearchDARTSEncoderCell, SearchGDASEncoderCell, SearchGDASDecoderCell, SearchDARTSDecoderCell
from tsf_oneshot.cells.ops import PRIMITIVES_Encoder
from tsf_oneshot.cells.encoders.components import _Chomp1d


class EmbeddingLayer(nn.Module):
    # https://github.com/cure-lab/LTSF-Linear/blob/main/layers/Embed.py
    def __init__(self, c_in, d_model, kernel_size=3):
        super(EmbeddingLayer, self).__init__()
        padding = (kernel_size - 1)
        self.tokenConv = nn.Conv1d(in_channels=c_in,
                                   out_channels=d_model,
                                   kernel_size=3, padding=padding,
                                   bias=False
                                   )
        self.chomp1 = _Chomp1d(padding)

    def forward(self, x_past: torch.Tensor):
        return self.chomp1(self.tokenConv(x_past.permute(0, 2, 1))).transpose(1, 2)


class AbstractSearchEncoder(nn.Module):
    def __init__(self,
                 d_input: int,
                 forecasting_horizon: int,
                 d_model: int,
                 n_cells: int,
                 n_nodes: int = 4,
                 n_cell_input_nodes: int = 1,
                 PRIMITIVES=PRIMITIVES_Encoder.keys(),
                 ):
        super(AbstractSearchEncoder, self).__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.n_cells = n_cells
        self.n_nodes = n_nodes
        self.ops = PRIMITIVES
        self.n_cell_input_nodes = n_cell_input_nodes

        #self.embedding_layer = nn.Linear(d_input, d_model, bias=True)
        self.embedding_layer = EmbeddingLayer(d_input, d_model)

        cells = []
        num_edges = None
        edge2index = None
        for i in range(n_cells):
            cell = self.get_cell(n_nodes=n_nodes,
                                 n_input_nodes=n_cell_input_nodes,
                                 forecasting_horizon=forecasting_horizon,
                                 d_model=d_model,
                                 PRIMITIVES=PRIMITIVES,
                                 is_first_cell=(i==0)
                                 )
            cells.append(cell)
            if num_edges is None:
                num_edges = cell.num_edges
                edge2index = cell.edge2index
        self.cells = nn.ModuleList(cells)

        self.num_edges = num_edges

        # self.arch_parameters = nn.Parameter(1e-3 * torch.randn(num_edges, len(PRIMITIVES)))

        self._device = torch.device('cpu')

    @staticmethod
    def get_cell(**kwargs):
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self.to(device)
        self._device = device
        # self.arch_parameters = self.arch_parameters.to(device)

    def save(self, base_path: Path):
        meta_info = {
            'input_dim': self.input_dims,
            'd_model': self.d_model,
            'n_cells': self.n_cells,
            'n_cell_input_nodes': self.n_cell_input_nodes,
            'PRIMITIVES': self.PRIMITIVES}

        if not base_path.exists():
            os.makedirs(base_path)

        with open(base_path / f'meta_info.json', 'w') as f:
            json.dump(meta_info, f)
        torch.save(self.state_dict(), base_path / f'model_weights.pth')

    @staticmethod
    def load(base_path: Path, model_type: Type["SearchEncoder"], device=torch.device('cpu')):
        with open(base_path / f'meta_info.json', 'r') as f:
            meta_info = json.load(f)
        model = model_type(**meta_info)
        if (base_path / f'model_weights.pth').exists():
            model.load_state_dict(torch.load(base_path / f'model_weights.pth', map_location=device))
        return model

    def forward(self, past_features: torch.Tensor, w_dag, **kwargs):
        raise NotImplementedError


class SearchDARTSEncoder(AbstractSearchEncoder):
    @staticmethod
    def get_cell(**kwargs):
        return SearchDARTSEncoderCell(**kwargs)

    def save(self, base_path: Path):
        meta_info = {
            'input_dim': self.input_dims,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_cell_input_nodes': self.n_cell_input_nodes,
            'PRIMITIVES': self.PRIMITIVES,
        }

        if not base_path.exists():
            os.makedirs(base_path)

        with open(base_path / f'meta_info.json', 'w') as f:
            json.dump(meta_info, f)
        torch.save(self.state_dict(), base_path / f'model_weights.pth')

    def forward(self, past_features: torch.Tensor, w_dag: torch.Tensor, **kwargs):
        embedding = self.embedding_layer(past_features)
        states = [embedding] * self.n_cell_input_nodes
        cell_out = None

        cell_intermediate_steps = []
        for cell in self.cells:
            cell_out = cell(s_previous=states, w_dag=w_dag, )
            states = [*states[1:], cell_out]
            cell_intermediate_steps.append(cell.intermediate_outputs)
        return cell_out, cell_intermediate_steps


class SearchGDASEncoder(AbstractSearchEncoder):
    @staticmethod
    def get_cell(**kwargs):
        return SearchGDASEncoderCell(**kwargs)

    def forward(self, past_features: torch.Tensor, w_dag: tuple[torch.Tensor, torch.Tensor], **kwargs):
        hardwts, index = w_dag
        past_features = self.embedding_layer(past_features)
        states = [past_features] * self.n_cell_input_nodes
        cell_out = None
        cell_intermediate_steps = []

        for cell in self.cells:
            cell_out = cell.forward_gdas(s_previous=states, w_dag=hardwts, index=index)
            states = [*states[1:], cell_out]
            cell_intermediate_steps.append(cell.intermediate_outputs)

        return cell_out, cell_intermediate_steps


class SearchDARTSDecoder(SearchDARTSEncoder):
    @staticmethod
    def get_cell(**kwargs):
        return SearchDARTSDecoderCell(**kwargs)

    def forward(self,
                future_features: torch.Tensor,
                w_dag: torch.Tensor,
                cells_encoder_output: list[torch.Tensor],
                net_encoder_output: torch.Tensor,
                ):
        embedding = self.embedding_layer(future_features)
        states = [embedding] * self.n_cell_input_nodes
        cell_out = None
        # w_dag = self.apply_normalizer(self.arch_parameters)
        # print(f'Decoder: {embedding.min()} and {embedding.max()}')

        for cell_encoder_output, cell in zip(cells_encoder_output, self.cells):
            cell_out = cell(s_previous=states, w_dag=w_dag,
                            cell_encoder_output=cell_encoder_output, net_encoder_output=net_encoder_output)
            states = [*states[1:], cell_out]
        return cell_out


class SearchGDASDecoder(SearchDARTSDecoder):
    @staticmethod
    def get_cell(**kwargs):
        return SearchGDASDecoderCell(**kwargs)

    def forward(self,
                future_features: torch.Tensor,
                w_dag: tuple[torch.Tensor, torch.Tensor],
                cells_encoder_output: list[torch.Tensor],
                net_encoder_output: torch.Tensor,
                ):
        hardwts, index = w_dag
        future_features = self.embedding_layer(future_features)
        states = [future_features] * self.n_cell_input_nodes
        cell_out = None
        for cell_encoder_output, cell in zip(cells_encoder_output, self.cells):
            cell_out = cell.forward_gdas(s_previous=states, w_dag=hardwts, index=index,
                                         cell_encoder_output=cell_encoder_output, net_encoder_output=net_encoder_output)
            states = [*states[1:], cell_out]

        return cell_out
