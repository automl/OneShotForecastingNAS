import copy
from pathlib import Path
from typing import Type
import os
import json

import torch
from torch import nn

from tsf_oneshot.cells.cells import (
    SearchDARTSEncoderCell,
    SearchGDASEncoderCell,
    SearchGDASDecoderCell,
    SearchDARTSDecoderCell,

    SearchDARTSFlatEncoderCell,
    SearchGDASFlatEncoderCell
)
from tsf_oneshot.cells.ops import PRIMITIVES_Encoder, PRIMITIVES_FLAT_ENCODER
from tsf_oneshot.cells.utils import EmbeddingLayer



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    from https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    from https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class AbstractSearchEncoder(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_model: int,
                 n_cells: int,
                 window_size: int,
                 forecasting_horizon: int,
                 n_nodes: int = 4,
                 n_cell_input_nodes: int = 1,
                 PRIMITIVES=PRIMITIVES_Encoder.keys(),
                 OPS_kwargs: dict[str, dict] = {}
                 ):
        super(AbstractSearchEncoder, self).__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.n_cells = n_cells
        self.n_nodes = n_nodes
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
                                 d_inputs=d_inputs,
                                 window_size=window_size,
                                 forecasting_horizon=forecasting_horizon,
                                 n_input_nodes=n_cell_input_nodes,
                                 d_model=d_model,
                                 PRIMITIVES=PRIMITIVES,
                                 cell_idx=i,
                                 OPS_kwargs=OPS_kwargs,
                                 )
            d_inputs = [*d_inputs[1:], d_model]
            cells.append(cell)
            if num_edges is None:
                num_edges = cell.num_edges
                edge2index = cell.edge2index
        self.cells = nn.ModuleList(cells)
        self.edge2index = edge2index

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
            'PRIMITIVES': self.ops,
            'OPS_kwargs': self.OPS_kwargs
        }

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

    def forward(self, x: torch.Tensor, **kwargs):
        if isinstance(x, torch.Tensor):
            states = [x for _ in range(self.n_cell_input_nodes)]
        elif isinstance(x, list):
            # TODO check the cases when len(states) != self.n_cell_input_nodes !!!
            states = x
        else:
            raise NotImplementedError(f'Unknown input type: {type(x)}')
        return self.cells_forward(states, **kwargs)

    def cells_forward(self, states, **kwargs):
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

    def cells_forward(self, states: list[torch.Tensor], w_dag: torch.Tensor, **kwargs):
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

    def cells_forward(self, states: list[torch.Tensor], w_dag: torch.Tensor, **kwargs):
        hardwts, index = w_dag
        cell_out = None
        cell_intermediate_steps = []
        for cell in self.cells:
            cell_out = cell.forward_gdas(s_previous=states, w_dag=hardwts, index=index)
            states = [*states[1:], cell_out]
            cell_intermediate_steps.append(cell.intermediate_outputs)
        return cell_out, cell_intermediate_steps


class SearchDARTSDecoder(SearchDARTSEncoder):
    def __init__(self, use_psec=True, **kwargs):
        super(SearchDARTSDecoder, self).__init__(**kwargs)

    @staticmethod
    def get_cell(**kwargs):
        return SearchDARTSDecoderCell(**kwargs)

    def cells_forward(self, states: list[torch.Tensor], w_dag: torch.Tensor,
                      cells_encoder_output: list[torch.Tensor],
                      net_encoder_output: torch.Tensor, **kwargs):
        cell_out = None
        for cell_encoder_output, cell in zip(cells_encoder_output, self.cells):
            cell_out = cell(s_previous=states, w_dag=w_dag,
                            cell_encoder_output=cell_encoder_output, net_encoder_output=net_encoder_output)
            states = [*states[1:], cell_out]
        return cell_out


class LinearDecoder(nn.Module):
    def __init__(self, window_size: int, forecasting_horizon, d_input_future:int, d_model:int, dropout: float=0.2):
        """
        A naive Linear decoder that maps the decoder to a simple linear layer
        :param window_size:
        :param forecasting_horizon:
        """
        super(LinearDecoder, self).__init__()
        self.window_size = window_size
        self.forecasting_horizon = forecasting_horizon
        self.d_input_future = d_input_future

        self.embedding_layer = EmbeddingLayer(d_input_future, d_model)
        #self.embedding_layer = nn.Linear(d_input_future, d_model)
        self.norm = nn.InstanceNorm1d(forecasting_horizon, affine=True, track_running_stats=False)
        #self.norm = nn.LayerNorm(d_model)
        #self.linear_decoder = nn.Linear(window_size + forecasting_horizon, forecasting_horizon)

        #self.linear_decoder = nn.Linear(window_size + forecasting_horizon, forecasting_horizon)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_decoder = nn.Sequential(
            #nn.Linear(window_size, forecasting_horizon),
            nn.Conv1d(window_size, forecasting_horizon, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            #self.norm,
        )

    def forward(self, x: torch.Tensor, net_encoder_output: torch.Tensor, **kwargs):
        #if isinstance(x, torch.Tensor):
        #    embedding = self.embedding_layer(x)
        #elif isinstance(x, list):
        #    # TODO check the cases when len(states) != self.n_cell_input_nodes !!!
        #    embedding = self.embedding_layer(x[-1])
        #else:
        #    raise NotImplementedError
        #net_encoder_output = torch.cat([net_encoder_output, embedding], dim=1)
        return self.linear_decoder(net_encoder_output)
        #return self.linear_decoder(net_encoder_output.permute(0, 2, 1)).permute(0, 2, 1)



class SearchGDASDecoder(SearchDARTSDecoder):
    def __init__(self, use_psec=True, **kwargs):
        super(SearchDARTSDecoder, self).__init__(**kwargs)

    @staticmethod
    def get_cell(**kwargs):
        return SearchGDASDecoderCell(**kwargs)

    def cells_forward(self, states: list[torch.Tensor], w_dag: torch.Tensor,
                      cells_encoder_output: list[torch.Tensor],
                      net_encoder_output: torch.Tensor, **kwargs):
        hardwts, index = w_dag
        cell_out = None
        for cell_encoder_output, cell in zip(cells_encoder_output, self.cells):
            cell_out = cell.forward_gdas(s_previous=states, w_dag=hardwts, index=index,
                                         cell_encoder_output=cell_encoder_output, net_encoder_output=net_encoder_output)
            states = [*states[1:], cell_out]
        return cell_out


class AbstractFlatEncoder(AbstractSearchEncoder):
    def __init__(self,
                 window_size: int,
                 forecasting_horizon: int,
                 d_output: int,
                 n_cells: int,
                 n_nodes: int = 4,
                 n_cell_input_nodes: int = 1,
                 PRIMITIVES=PRIMITIVES_FLAT_ENCODER.keys(),
                 OPS_kwargs: dict[str, dict] = {}
                 ):
        self.window_size = window_size
        self.forecasting_horizon = forecasting_horizon
        self.n_cells = n_cells
        self.n_nodes = n_nodes
        self.n_cells_input_nodes = n_cell_input_nodes
        self.ops = PRIMITIVES
        self.OPS_kwargs = OPS_kwargs

        self.d_output = d_output

        nn.Module.__init__(self)
        cells = []
        num_edges = None
        edge2index = None
        for i in range(n_cells):
            cell = self.get_cell(window_size=window_size,
                                 n_input_nodes=n_cell_input_nodes,
                                 n_nodes=n_nodes,
                                 forecasting_horizon=forecasting_horizon,
                                 PRIMITIVES=PRIMITIVES,
                                 is_last_cell=(i == n_cells - 1),
                                 OPS_kwargs=OPS_kwargs,
                                 )
            cells.append(cell)
            if num_edges is None:
                num_edges = cell.num_edges
                edge2index  = cell.edge2index
        self.cells = nn.ModuleList(cells)
        self.edge2index = edge2index

        self.num_edges = num_edges

        self._device = torch.device('cpu')

    def save(self, base_path: Path):
        meta_info = {
            'input_dim': self.input_dims,
            'window_size': self.window_size,
            'forecasting_horizon': self.forecasting_horizon,
            'n_cells': self.n_cells,
            'n_cell_input_nodes': self.n_cell_input_nodes,
            'PRIMITIVES': self.ops,
            'OPS_kwargs': self.OPS_kwargs
        }

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


class SearchDARTSFlatEncoder(AbstractFlatEncoder):
    def __init__(self, **kwargs):
        super(SearchDARTSFlatEncoder, self).__init__(**kwargs)
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

    @staticmethod
    def get_cell(**kwargs):
        return SearchDARTSFlatEncoderCell(**kwargs)

    def forward(self, x: torch.Tensor, w_dag: torch.Tensor, **kwargs):
        # we always transform the input multiple_series map into independent single series input

        x = x[:, :, :self.d_output]

        #"""
        # This result in a feature map of size [B, N, L, 1]
        past_targets = torch.transpose(x, -1, -2)

        future_targets = torch.zeros([*past_targets.shape[:-1], self.forecasting_horizon], device=past_targets.device,
                                     dtype=past_targets.dtype)
        embedding = torch.cat(
            [past_targets, future_targets], dim=-1
        )
        states = [embedding]
        """
        seasonal_init, trend_init = self.decompsition(x)
        # This result in a feature map of size [B*N, L, 1]
        #past_targets = torch.transpose(x_past, -1, -2)
        x_s = torch.transpose(seasonal_init, -1, -2)
        x_t = torch.transpose(trend_init, -1, -2)
        future_targets = torch.zeros([*x_s.shape[:-1], self.forecasting_horizon], device=x_s.device,
                                     dtype=x_s.dtype)
        embedding_s = torch.cat([x_s, future_targets], dim=-1)
        embedding_t = torch.cat([x_t, future_targets], dim=-1)

        states = [embedding_s, embedding_t]
        #"""

        cell_out = self.cells_forward(states, w_dag)
        cell_out = cell_out.transpose(-1, -2)
        return cell_out

    def cells_forward(self, states: list[torch], w_dag: torch.Tensor, **kwargs):
        cell_out = None
        for cell in self.cells:
            cell_out = cell(s_previous=states, w_dag=w_dag, )
            states = [*states[1:], cell_out]
        return cell_out


class SearchGDASFlatEncoder(SearchDARTSFlatEncoder):
    @staticmethod
    def get_cell(**kwargs):
        return SearchGDASFlatEncoderCell(**kwargs)

    def cells_forward(self, states: list[torch], w_dag: torch.Tensor, **kwargs):
        cell_out = None
        hardwts, index = w_dag
        for cell in self.cells:
            cell_out = cell.forward_gdas(s_previous=states, w_dag=hardwts, index=index, )
            states = [*states[1:], cell_out]
        return cell_out

