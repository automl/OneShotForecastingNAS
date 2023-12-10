import torch
from torch import nn

from tsf_oneshot.networks.components import SearchGDASEncoder, SearchGDASDecoder, SearchDARTSEncoder, SearchDARTSDecoder
from tsf_oneshot.prediction_heads import MixedHead
import wandb

class ForecastingAbstractNetwork(nn.Module):
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
                 HEADS_kwargs: dict[str, dict],
                 ):
        super(ForecastingAbstractNetwork, self).__init__()
        self.d_input_past = d_input_past
        self.d_input_future = d_input_future
        self.d_output = d_output
        self.d_model = d_model

        self.n_cells = n_cells
        self.n_nodes = n_nodes
        self.n_cell_input_nodes = n_cell_input_nodes
        self.PRIMITIVES_encoder = PRIMITIVES_encoder
        self.PRIMITIVES_decoder = PRIMITIVES_decoder
        self.HEADS = HEADS

        encoder_kwargs = dict(d_input=d_input_past,
                              d_model=d_model,
                              n_cells=n_cells,
                              n_nodes=n_nodes,
                              n_cell_input_nodes=n_cell_input_nodes,
                              PRIMITIVES=PRIMITIVES_encoder,
                              forecasting_horizon=forecasting_horizon,
                              )

        self.encoder: SearchDARTSEncoder = self.get_encoder(
            **encoder_kwargs
        )

        decoder_kwargs = dict(d_input=d_input_future,
                              d_model=d_model,
                              n_cells=n_cells,
                              n_nodes=n_nodes,
                              n_cell_input_nodes=n_cell_input_nodes,
                              PRIMITIVES=PRIMITIVES_decoder,
                              forecasting_horizon=forecasting_horizon
                              )

        self.decoder: SearchDARTSDecoder = self.get_decoder(
            **decoder_kwargs
        )

        self.heads = MixedHead(d_model=d_model, d_output=d_output, PRIMITIVES=HEADS, OPS_kwargs=HEADS_kwargs)

        self.encoder_n_edges = self.encoder.num_edges
        self.decoder_n_edges = self.decoder.num_edges

    @staticmethod
    def get_encoder(**encoder_kwargs):
        raise NotImplementedError

    @staticmethod
    def get_decoder(**decoder_kwargs):
        raise NotImplementedError

    def get_head_out(self, cell_decoder_out: torch.Tensor, head_idx: int | None=None):
        if head_idx is None:
            return list(head(cell_decoder_out) for head in self.heads)
        else:
            return self.heads[head_idx](cell_decoder_out)

    def forward(self, x_past: torch.Tensor,
                x_future: torch.Tensor,
                arch_p_encoder: torch.Tensor,
                arch_p_decoder: torch.Tensor,
                arch_p_heads: torch.Tensor):
        cell_encoder_out, cell_intermediate_steps = self.encoder(x_past, w_dag=arch_p_encoder)

        cell_decoder_out = self.decoder(future_features=x_future,
                                        w_dag=arch_p_decoder,
                                        cells_encoder_output=cell_intermediate_steps,
                                        net_encoder_output=cell_encoder_out)
        return self.get_head_out(cell_decoder_out, head_idx=None)


class ForecastingDARTSNetwork(ForecastingAbstractNetwork):
    @staticmethod
    def get_encoder(**encoder_kwargs):
        return SearchDARTSEncoder(**encoder_kwargs)

    @staticmethod
    def get_decoder(**decoder_kwargs):
        return SearchDARTSDecoder(**decoder_kwargs)


class ForecastingGDASNetwork(ForecastingAbstractNetwork):
    @staticmethod
    def get_encoder(**encoder_kwargs):
        return SearchGDASEncoder(**encoder_kwargs)

    @staticmethod
    def get_decoder(**decoder_kwargs):
        return SearchGDASDecoder(**decoder_kwargs)

    def forward(self, x_past: torch.Tensor,
                x_future: torch.Tensor,
                arch_p_encoder: tuple[torch.Tensor, torch.Tensor],
                arch_p_decoder: tuple[torch.Tensor, torch.Tensor],
                arch_p_heads: tuple[torch.Tensor, torch.Tensor]):

        cell_encoder_out, cell_intermediate_steps = self.encoder(x_past, w_dag=arch_p_encoder,)

        cell_decoder_out = self.decoder(future_features=x_future,
                                        w_dag=arch_p_decoder,
                                        cells_encoder_output=cell_intermediate_steps,
                                        net_encoder_output=cell_encoder_out)
        return self.get_head_out(cell_decoder_out, head_idx=None)

