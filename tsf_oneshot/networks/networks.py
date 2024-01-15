import inspect
import torch
from torch import nn

from tsf_oneshot.networks.components import (
    SearchGDASEncoder,
    SearchGDASDecoder,
    SearchGDASFlatEncoder,
    SearchDARTSFlatEncoder,
    SearchDARTSEncoder,
    SearchDARTSDecoder,
    LinearDecoder
)
from tsf_oneshot.prediction_heads import MixedHead, MixedFlatHEADAS
from tsf_oneshot.networks.utils import get_head_out
from tsf_oneshot.networks.combined_net_utils import (
    get_kwargs,
    forward_concat_net,
    forward_parallel_net
)

DECODER_MAPS = {
    'seq': 'seq_decoder',
    'linear': 'linear_decoder'
}


class ForecastingAbstractNetwork(nn.Module):
    def __init__(self,
                 d_input_past: int,
                 d_input_future: int,
                 window_size: int,
                 forecasting_horizon: int,
                 d_model: int,
                 d_output: int,
                 n_cells: int,
                 n_nodes: int,
                 n_cell_input_nodes: int,
                 PRIMITIVES_encoder: list[str],
                 PRIMITIVES_decoder: list[str],
                 OPS_kwargs: dict[str, dict],
                 HEADs: list[str],
                 HEADs_kwargs: dict[str, dict],
                 DECODERS: list[str] = ['seq'],
                 decoder_use_psec: bool=False,
                 forecast_only: bool = False
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
        self.HEADs = HEADs
        self.forecast_only = forecast_only

        self.DECODERS = DECODERS

        self.window_size = window_size
        self.forecasting_horizon = forecasting_horizon

        encoder_kwargs = dict(d_input=d_input_past,
                              d_model=d_model,
                              n_cells=n_cells,
                              n_nodes=n_nodes,
                              n_cell_input_nodes=n_cell_input_nodes,
                              PRIMITIVES=PRIMITIVES_encoder,
                              OPS_kwargs=OPS_kwargs,
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
                              use_psec=decoder_use_psec,
                              OPS_kwargs=OPS_kwargs
                              )
        if 'seq' in DECODERS:
            self.seq_decoder: SearchDARTSDecoder = self.get_decoder(
                **decoder_kwargs
            )
        if 'linear' in DECODERS:
            self.linear_decoder: LinearDecoder = LinearDecoder(window_size, forecasting_horizon,
                                                               d_input_future=d_input_future,
                                                               d_model=d_model)
        self.decoder = [getattr(self, DECODER_MAPS[decoder]) for decoder in DECODERS]

        self.heads = MixedHead(d_model=d_model, d_output=d_output, PRIMITIVES=HEADs, OPS_kwargs=HEADs_kwargs)

        self.n_cell_nodes = n_nodes + n_cell_input_nodes
        self.n_edges = self.encoder.num_edges

        self.edge2index = self.encoder.edge2index

    @staticmethod
    def get_encoder(**encoder_kwargs):
        raise NotImplementedError

    @staticmethod
    def get_decoder(**decoder_kwargs):
        raise NotImplementedError

    def get_decoder_out(self,
                        arch_p_decoder_choices: torch.Tensor,
                        decoder_forward_kwargs: dict):
        raise NotImplementedError

    def forward(self, x_past: torch.Tensor,
                x_future: torch.Tensor,
                arch_p_encoder: torch.Tensor,
                arch_p_decoder: torch.Tensor,
                arch_p_heads: torch.Tensor,
                arch_p_decoder_choices):
        net_encoder_out, cell_intermediate_steps = self.encoder(x_past, w_dag=arch_p_encoder)
        decoder_forward_kwargs = dict(x=x_future,
                                      w_dag=arch_p_decoder,
                                      cells_encoder_output=cell_intermediate_steps,
                                      net_encoder_output=net_encoder_out)

        cell_decoder_out = self.get_decoder_out(arch_p_decoder_choices=arch_p_decoder_choices[0],
                                                decoder_forward_kwargs=decoder_forward_kwargs)

        forecast = get_head_out(cell_decoder_out, heads=self.heads, head_idx=None)

        if self.forecast_only:
            return forecast
        backcast = get_head_out(net_encoder_out, heads=self.heads, head_idx=None)
        return backcast, forecast


class ForecastingDARTSNetwork(ForecastingAbstractNetwork):
    @staticmethod
    def get_encoder(**encoder_kwargs):
        return SearchDARTSEncoder(**encoder_kwargs)

    @staticmethod
    def get_decoder(**decoder_kwargs):
        return SearchDARTSDecoder(**decoder_kwargs)

    def get_decoder_out(self,
                        arch_p_decoder_choices: torch.Tensor,
                        decoder_forward_kwargs: dict
                        ):
        return sum(w * de(**decoder_forward_kwargs) for w, de in zip(arch_p_decoder_choices, self.decoder) if w > 0.)


class ForecastingGDASNetwork(ForecastingAbstractNetwork):
    @staticmethod
    def get_encoder(**encoder_kwargs):
        return SearchGDASEncoder(**encoder_kwargs)

    @staticmethod
    def get_decoder(**decoder_kwargs):
        return SearchGDASDecoder(**decoder_kwargs)

    def get_decoder_out(self,
                        arch_p_decoder_choices: torch.Tensor,
                        decoder_forward_kwargs: dict
                        ):
        hardwts, index = arch_p_decoder_choices
        argmaxs = index.item()

        weigsum = sum(
            hardwts[_ie] for _ie in range(self.n_ops) if _ie != argmaxs
        )
        cell_decoder_out = self.decoder[argmaxs](**decoder_forward_kwargs) + weigsum
        return cell_decoder_out


class ForecastingFlatAbstractNetwork(nn.Module):
    def __init__(self,
                 window_size: int,
                 forecasting_horizon: int,
                 d_output: int,
                 n_cells: int,
                 n_nodes: int,
                 n_cell_input_nodes: int,
                 PRIMITIVES_encoder: list[str],
                 OPS_kwargs: dict[str, dict],
                 HEADs: list[str],
                 HEADs_kwargs: dict[str, dict],
                 forecast_only: bool = False
                 ):
        super(ForecastingFlatAbstractNetwork, self).__init__()
        self.window_size = window_size
        self.forecasting_horizon = forecasting_horizon
        self.n_cells = n_cells
        self.n_nodes = n_nodes
        self.n_cell_input_node = n_cell_input_nodes

        self.PRIMITIVES_encoder = PRIMITIVES_encoder
        self.OPS_kwargs = OPS_kwargs
        self.HEADs = HEADs
        self.HEADs_kwargs = HEADs_kwargs
        self.forecast_only = forecast_only
        self.d_output = d_output

        encoder_kwargs = dict(window_size=window_size,
                              forecasting_horizon=forecasting_horizon,
                              d_output=d_output,
                              n_cells=n_cells,
                              n_nodes=n_nodes,
                              n_cell_input_nodes=n_cell_input_nodes,
                              PRIMITIVES=PRIMITIVES_encoder,
                              OPS_kwargs=OPS_kwargs,
                              )

        self.encoder = self.get_encoder(**encoder_kwargs)
        self.heads = MixedFlatHEADAS(window_size=window_size, forecasting_horizon=forecasting_horizon,
                                     PRIMITIVES=HEADs, OPS_kwargs=HEADs_kwargs)

        self.n_cell_nodes = n_nodes + n_cell_input_nodes
        self.n_edges = self.encoder.num_edges
        self.edge2index = self.encoder.edge2index

    @staticmethod
    def get_encoder(**encoder_kwargs):
        return SearchDARTSFlatEncoder(**encoder_kwargs)

    def forward(self, x_past: torch.Tensor,
                x_future: torch.Tensor,
                arch_p_encoder: torch.Tensor,
                arch_p_heads: tuple[torch.Tensor, torch.Tensor],
                forward_only_with_net: bool = False,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        cell_decoder_out = self.encoder(x_past, w_dag=arch_p_encoder)

        backcast, forecast = torch.split(cell_decoder_out, [self.window_size, self.forecasting_horizon], dim=1)
        if forward_only_with_net:
            if self.forecast_only:
                return forecast
            return backcast, forecast
        forecast = get_head_out(forecast, heads=self.heads, head_idx=None)
        if self.forecast_only:
            return forecast
        backcast = get_head_out(backcast, heads=self.heads, head_idx=None)
        return backcast, forecast

    def get_head_out(self, cell_decoder_out: torch.Tensor, head_idx: int | None = None):
        if head_idx is None:
            return list(head(cell_decoder_out) for head in self.heads)
        else:
            return self.heads[head_idx](cell_decoder_out)


class ForecastingDARTSFlatNetwork(ForecastingFlatAbstractNetwork):
    @staticmethod
    def get_encoder(**encoder_kwargs):
        return SearchDARTSFlatEncoder(**encoder_kwargs)


class ForecastingGDASFlatNetwork(ForecastingFlatAbstractNetwork):
    @staticmethod
    def get_encoder(**encoder_kwargs):
        return SearchGDASFlatEncoder(**encoder_kwargs)


class ForecastingAbstractMixedNet(nn.Module):
    decoder_use_psec_seq = False

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
                 DECODERS_seq: list[str],

                 OPS_kwargs_seq: dict[str, dict],
                 HEADs: list[str],
                 HEADs_kwargs_seq: dict[str, dict],
                 forecast_only_seq: bool,

                 window_size: int,
                 forecasting_horizon: int,
                 n_cells_flat: int,
                 n_nodes_flat: int,
                 n_cell_input_nodes_flat: int,
                 PRIMITIVES_encoder_flat: list[str],
                 OPS_kwargs_flat: dict[str, dict],
                 HEADs_kwargs_flat: dict[str, dict],
                 forecast_only_flat: bool = True

                 ):
        all_kwargs = get_kwargs()
        nn.Module.__init__(self)

        self.meta_info = all_kwargs

        all_kwargs = self.validate_input_kwargs(all_kwargs)

        # get arguments for the seq net
        seq_net_kwargs = {}
        for arg_name in inspect.signature(ForecastingAbstractNetwork.__init__).parameters:
            if arg_name != 'self':
                if arg_name in all_kwargs:
                    seq_net_kwargs[arg_name] = all_kwargs[arg_name]
                elif f'{arg_name}_seq' in all_kwargs:
                    seq_net_kwargs[arg_name] = all_kwargs[f'{arg_name}_seq']

        self.seq_net = self.get_seq_net(decoder_use_psec=self.decoder_use_psec_seq, **seq_net_kwargs)

        # get arguments for the flat net
        flat_net_kwargs = {}
        for arg_name in inspect.signature(ForecastingFlatAbstractNetwork.__init__).parameters:
            if arg_name != 'self':
                if arg_name in all_kwargs:
                    flat_net_kwargs[arg_name] = all_kwargs[arg_name]
                else:
                    flat_net_kwargs[arg_name] = all_kwargs[f'{arg_name}_flat']

        self.flat_net = self.get_flat_net(**flat_net_kwargs)

        # this function helps us to directly
        self.d_output = d_output

        # TODO check how to properly handle the heads
        self.heads = self.seq_net.heads

        # for the case of concat output, flat
        self.forecast_only_flat = forecast_only_flat
        self.forecast_only_seq = forecast_only_seq

        self.n_cell_nodes_seq = n_nodes_seq + n_cell_input_nodes_seq
        self.n_cell_nodes_flat = n_nodes_flat + n_cell_input_nodes_flat

        self.n_cell_input_nodes_seq = n_cell_input_nodes_seq
        self.n_cell_input_nodes_flat = n_cell_input_nodes_flat

        self.edge2index_seq = self.seq_net.edge2index
        self.edge2index_flat = self.flat_net.edge2index

    def validate_input_kwargs(self, kwargs):
        return kwargs

    def get_net_out(self, **kwargs):
        raise NotImplementedError

    def get_seq_forward_kwargs(self, arch_p_encoder_seq: torch.Tensor,
                               arch_p_decoder_seq: torch.Tensor,
                               arch_p_heads_seq: torch.Tensor,
                               arch_p_decoder_choices_seq):
        return dict(
            arch_p_encoder=arch_p_encoder_seq,
            arch_p_decoder=arch_p_decoder_seq,
            arch_p_heads=arch_p_heads_seq,
            arch_p_decoder_choices=arch_p_decoder_choices_seq
        )

    def get_flat_forward_kwargs(self,
                                arch_p_encoder_flat: torch.Tensor,
                                arch_p_heads_flat: torch.Tensor):
        return dict(
            arch_p_encoder=arch_p_encoder_flat,
            arch_p_heads=arch_p_heads_flat
        )

    def forward(self,
                x_past: torch.Tensor,
                x_future: torch.Tensor,
                arch_p_encoder_seq: torch.Tensor,
                arch_p_decoder_seq: torch.Tensor,
                arch_p_heads_seq: torch.Tensor,
                arch_p_encoder_flat: torch.Tensor,
                arch_p_heads_flat: torch.Tensor,
                arch_p_decoder_choices_seq: torch.Tensor,
                arch_p_net: torch.Tensor | None = None
                ):
        seq_kwargs = self.get_seq_forward_kwargs(
            arch_p_encoder_seq, arch_p_decoder_seq, arch_p_heads_seq, arch_p_decoder_choices_seq
        )

        flat_kwargs = self.get_flat_forward_kwargs(arch_p_encoder_flat, arch_p_heads_flat)
        return self.get_net_out(flat_net=self.flat_net,
                                seq_net=self.seq_net,
                                x_past=x_past,
                                x_future=x_future,
                                forecast_only_seq=self.forecast_only_seq,
                                forecast_only_flat=self.forecast_only_flat,
                                seq_kwargs=seq_kwargs,
                                flat_kwargs=flat_kwargs,
                                out_weights=arch_p_net[0],
                                )


class DARTSMixedNetMixin:
    @staticmethod
    def get_seq_net(**kwargs):
        return ForecastingDARTSNetwork(**kwargs)

    @staticmethod
    def get_flat_net(**kwargs):
        return ForecastingDARTSFlatNetwork(**kwargs)


class GDASMixedNetMixin:
    @staticmethod
    def get_seq_net(**kwargs):
        return ForecastingGDASNetwork(**kwargs)

    @staticmethod
    def get_flat_net(**kwargs):
        return ForecastingGDASFlatNetwork(**kwargs)


class ForecastingAbstractMixedParallelNet(ForecastingAbstractMixedNet):
    def get_net_out(self, **kwargs):
        return forward_parallel_net(**kwargs)


class ForecastingAbstractMixedConcatNet(ForecastingAbstractMixedNet):
    decoder_use_psec_seq = False

    def validate_input_kwargs(self, kwargs):
        kwargs['forecast_only_flat'] = False
        assert kwargs['d_input_future'] == kwargs['d_input_past']
        return kwargs

    def get_net_out(self, **kwargs):
        return forward_concat_net(**kwargs)


class ForecastingDARTSMixedParallelNet(DARTSMixedNetMixin, ForecastingAbstractMixedParallelNet):
    pass


class ForecastingGDASMixedParallelNet(GDASMixedNetMixin, ForecastingAbstractMixedParallelNet):
    pass


class ForecastingDARTSMixedConcatNet(DARTSMixedNetMixin, ForecastingAbstractMixedConcatNet):
    pass


class ForecastingGDASMixedConcatNet(GDASMixedNetMixin, ForecastingAbstractMixedConcatNet):
    pass
