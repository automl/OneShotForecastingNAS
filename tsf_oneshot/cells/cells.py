import copy

from typing import Callable

import torch
from torch import nn

from tsf_oneshot.cells.encoders import PRIMITIVES_Encoder, PRIMITIVES_FLAT_ENCODER
from tsf_oneshot.cells.decoders import PRIMITIVES_Decoder
from tsf_oneshot.cells.ops import MixedEncoderOps, MixedDecoderOps, MixedFlatEncoderOps


class AbstractSearchEncoderCell(nn.Module):
    op_types = MixedEncoderOps
    """Cell for searchs
    Each edge is mixed and continuous relaxed.

    Attributes:
        dag: List of lists where the out list corresponds to intermediate nodes in a cell. The inner
            list contains the mixed operations for each input node of an intermediate node (i.e.
            dag[i][j] calculates the outputs of the i-th intermediate node for its j-th input).
        preproc0: Preprocessing operation for the s0 input
        preproc1: Preprocessing operation for the s1 input
    """

    def __init__(self,
                 n_nodes: int,
                 d_model: int,
                 n_input_nodes: int = 1,
                 PRIMITIVES: list[str] = PRIMITIVES_Encoder.keys(),
                 OPS_kwargs: dict[str, dict] = {},
                 is_first_cell: bool = False,
                 aggregtrate_type: str = 'mean',
                 **kwargs,
                 ):
        """
        Args:
            n_nodes (int): Number of intermediate nodes. The output of the cell is calculated by
                concatenating the outputs of all intermediate nodes in the cell.
            d_model (int): model dimensions
            PRIMITIVES (list[str]): a list of all possible operations in this cells (the top one-shot model)
        """
        super().__init__()
        self.max_nodes = n_nodes + n_input_nodes
        self.n_input_nodes = n_input_nodes

        self.d_model = d_model
        self.op_names_all = copy.deepcopy(PRIMITIVES)
        # self.op_names_current = copy.deepcopy(PRIMITIVES_current)
        self.n_ops = len(self.op_names_all)

        # generate dag
        self.edges = nn.ModuleDict()
        self.aggregtrate_type = aggregtrate_type

        for i in range(n_input_nodes, self.max_nodes):
            # The first 2 nodes are input nodes
            for j in range(i):
                dilation = int(2 ** (i - 1))
                if 'tcn' in OPS_kwargs:
                    OPS_kwargs['tcn'].update({'dilation': dilation})
                else:
                    OPS_kwargs.update({'tcn': {'dilation': dilation}})

                if is_first_cell:
                    if i == n_input_nodes and j == 0:
                        if 'transformer' in OPS_kwargs:
                            OPS_kwargs['transformer'].update(
                                {'is_first_layer': True}
                            )
                        else:
                            OPS_kwargs.update({'transformer': {'is_first_layer': True}})
                node_str = f"{i}<-{j}"
                op = self.op_types(self.d_model,
                                   PRIMITIVES=PRIMITIVES,
                                   OPS_kwargs=OPS_kwargs)  # TODO check if PRIMITIVES fits the requirements?
                self.edges[node_str] = op

        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

        self.intermediate_outputs = {
            key: None for key in self.edge_keys
        }

    def forward(
            self, s_previous: list[torch.Tensor], w_dag: list[list[float]],
            edge_output_func: Callable,
            alpha_prune_threshold=0.0, **kwargs
    ):
        """Forward pass through the cell

        Args:
            s_previous: Output of the previous cells
            w_dag ( list[list[float]]): MixedOp weights ("alphas") (e.g. for n nodes and k primitive operations should
                be a list of length `n` of parameters where the n-th parameter has shape
                :math:`(n+2)xk = (number of inputs to the node) x (primitive operations)`)
                Each element in w_dag corresponds to an edge stored by self.edges
            edge_output_func (Callable): how to get the edge output
            alpha_prune_threshold:

        Returns:
            The output tensor of the cell
        """
        states = copy.copy(s_previous)

        for i in range(self.n_input_nodes, self.max_nodes):
            s_curs = []
            for j in range(i):
                node_str = f"{i}<-{j}"
                if node_str in self.edges:
                    edge_out = edge_output_func(node_str=node_str, x=states[j],
                                                w_dag=w_dag, alpha_prune_threshold=alpha_prune_threshold,
                                                **kwargs)
                    if isinstance(edge_out, torch.Tensor):
                        s_curs.append(edge_out)
            states.append(self.aggregate_edges_outputs(s_curs))

        return self.process_output(states)

    def aggregate_edges_outputs(self, s_curs: list[torch.Tensor]):
        return sum(s_curs)

    def get_edge_out(self, node_str, x, w_dag, alpha_prune_threshold, **kwargs):
        raise NotImplementedError

    def process_output(self, state: list[torch.Tensor]):
        return state[-1]

    def generate_edge_info(self):
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def extra_repr(self):
        string = "info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__
        )
        return string

    def get_meta_info(self) -> dict:
        """Get the meta info of the model that allows the users to reinitialize a new cell from this operations """
        meta_info = {
            "n_nodes": self.max_nodes - 2,
            "n_input_nodes": self.n_input_nodes,
            "d_model": self.d_model,
            "PRIMITIVES": self.op_names_all,
        }
        return meta_info

    def get_output_for_empty(self, x: torch.Tensor):
        out = torch.zeros_like(x)
        hx = out[:, [-1]].transpose(0, 1)
        return [out, hx, hx]


class SearchDARTSEncoderCell(AbstractSearchEncoderCell):
    aggregtrate_type = 'cat'
    """Cell for searchs
    Each edge is mixed and continuous relaxed.

    Attributes:
        dag: List of lists where the out list corresponds to intermediate nodes in a cell. The inner
            list contains the mixed operations for each input node of an intermediate node (i.e.
            dag[i][j] calculates the outputs of the i-th intermediate node for its j-th input).
        preproc0: Preprocessing operation for the s0 input
        preproc1: Preprocessing operation for the s1 input
    """

    def forward(
            self, s_previous: list[torch.Tensor], w_dag: list[list[float]],
            alpha_prune_threshold=0.0, **kwargs
    ):
        """Forward pass through the cell

        Args:
            s_previous: Output of the previous cells
            w_dag ( list[list[float]]): MixedOp weights ("alphas") (e.g. for n nodes and k primitive operations should
                be a list of length `n` of parameters where the n-th parameter has shape
                :math:`(n+2)xk = (number of inputs to the node) x (primitive operations)`)
                Each element in w_dag corresponds to an edge stored by self.edges
            alpha_prune_threshold:

        Returns:
            The output tensor of the cell
        """
        return super(SearchDARTSEncoderCell, self).forward(s_previous=s_previous, w_dag=w_dag,
                                                           alpha_prune_threshold=alpha_prune_threshold,
                                                           edge_output_func=self.get_edge_out, **kwargs)

    def get_edge_out(self, node_str, x, w_dag, alpha_prune_threshold, **kwargs):
        weights = w_dag[self.edge2index[node_str]]
        if torch.all(weights <= alpha_prune_threshold):
            self.intermediate_outputs[node_str] = self.get_output_for_empty(x)
            return []

        edge_out = self.edges[node_str](x_past=x, weights=weights,
                                        alpha_prune_threshold=alpha_prune_threshold)
        self.intermediate_outputs[node_str] = edge_out
        return edge_out[0]


class SearchDARTSDecoderCell(SearchDARTSEncoderCell):
    op_types = MixedDecoderOps
    """Cell for searchs
    Each edge is mixed and continuous relaxed.

    Attributes:
        dag: List of lists where the out list corresponds to intermediate nodes in a cell. The inner
            list contains the mixed operations for each input node of an intermediate node (i.e.
            dag[i][j] calculates the outputs of the i-th intermediate node for its j-th input).
        preproc0: Preprocessing operation for the s0 input
        preproc1: Preprocessing operation for the s1 input
    """

    def get_edge_out(self, node_str, x, w_dag, alpha_prune_threshold,
                     cell_encoder_output, net_encoder_output):
        encoder_out_edge = cell_encoder_output[node_str]
        edge_out = self.edges[node_str](x_future=x,
                                        encoder_output_net=net_encoder_output,
                                        encoder_output_layer=encoder_out_edge[0],
                                        hx1=encoder_out_edge[1],
                                        hx2=encoder_out_edge[2], weights=w_dag[self.edge2index[node_str]],
                                        alpha_prune_threshold=alpha_prune_threshold)
        return edge_out


class SearchGDASEncoderCell(AbstractSearchEncoderCell):
    """
    GDAS Search space, based on https://github.com/D-X-Y/AutoDL-Projects
    We adjust its architecture to be compatible with the DARTSCell
    """

    def process_output(self, state):
        return state[-1]

    def forward_gdas(self, s_previous: list[torch.Tensor], w_dag: list[list[float]], index, **kwargs):
        return super(SearchGDASEncoderCell, self).forward(s_previous=s_previous,
                                                          w_dag=w_dag, index=index,
                                                          edge_output_func=self.get_edge_out_gdas,
                                                          **kwargs
                                                          )

    def get_edge_out(self, node_str, x, w_dag, index, alpha_prune_threshold, **kwargs):
        return self.get_edge_out_gdas(node_str, x, w_dag)

    def get_edge_out_gdas(self, node_str, x, w_dag, index, **kwargs):
        edge_idx = self.edge2index[node_str]
        weights = w_dag[edge_idx]

        if torch.all(weights <= 0.0):
            self.intermediate_outputs[node_str] = self.get_output_for_empty(x)
            return []

        argmaxs = index[edge_idx].item()
        edge = self.edges[node_str]
        edge_out = edge[argmaxs](x_past=x)
        weigsum = sum(
            weights[_ie] for _ie in range(self.n_ops) if _ie != argmaxs
        )
        edge_out = [weights[argmaxs] * out + weigsum for out in edge_out]
        self.intermediate_outputs[node_str] = edge_out
        return edge_out[0]

    # GDAS Variant: https://github.com/D-X-Y/AutoDL-Projects/issues/119
    def forward_gdas_v1(self, s_previous: list[torch.Tensor], w_dag: list[list[float]], index, **kwargs):
        return super(SearchGDASEncoderCell, self).forward(s_previous=s_previous, w_dag=w_dag, index=index,
                                                          edge_output_func=self.get_edge_out_gdasv1,
                                                          **kwargs
                                                          )

    def get_edge_out_gdasv1(self, node_str, x, w_dag, index, **kwargs):
        weights = w_dag[self.edge2index[node_str]]

        if torch.all(weights <= 0.0):
            self.intermediate_outputs[node_str] = self.get_output_for_empty(x)
            return []

        argmaxs = index[self.edge2index[node_str]].item()
        edge_out = self.edges[node_str][argmaxs](x_past=x)
        edge_out = [weights[argmaxs] * out for out in edge_out]
        self.intermediate_outputs[node_str] = edge_out
        return edge_out[0]

    """
    # joint
    def forward_joint(self, s0: torch.Tensor, s1: torch.Tensor, w_dag: list[list[float]], index):
        states = self.input_preprocessing(s0, s1)

        for i in range(2, self.max_nodes):
            s_cur = 0.
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                if node_str in self.edges:
                    weights = w_dag[self.edge2index[node_str]]
                    # aggregation = sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) / weights.numel()
                    aggregation = sum(
                        layer(states[j]) * w
                        for layer, w in zip(self.edges[node_str], weights)
                    )
                    s_cur += aggregation
            states.append(s_cur)
        return self.process_output(states)


    # uniform random sampling per iteration, SETN
    def forward_urs(self, s0: torch.Tensor, s1: torch.Tensor):
        states = self.input_preprocessing(s0, s1)

        for i in range(2, self.max_nodes):
            while True:  # to avoid select zero for all ops
                sops, has_non_zero = [], False
                for j in range(i):
                    node_str = "{:}<-{:}".format(i, j)
                    if node_str in self.edges:
                        candidates = self.edges[node_str]
                        select_idx = torch.randint(len(candidates))
                        select_op = candidates[select_idx]
                        sops.append(select_op)
                        if not hasattr(select_op, "is_zero") or select_op.is_zero is False:
                            has_non_zero = True
                if has_non_zero:
                    break
            s_cur = 0.
            for j, select_op in enumerate(sops):
                s_cur += select_op(states[j])
            states.append(s_cur)
        return self.process_output(states)
    """

    # select the argmax
    def forward_select(self, s_previous: list[torch.Tensor], w_dag, **kwargs):
        return super(SearchGDASEncoderCell, self).forward(s_previous=s_previous, w_dag=w_dag,
                                                          edge_output_func=self.get_edge_out_select,
                                                          **kwargs
                                                          )

    def get_edge_out_select(self, node_str, x, w_dag, index, **kwargs):
        weights = w_dag[self.edge2index[node_str]]

        if torch.all(weights <= 0.0):
            self.intermediate_outputs[node_str] = self.get_output_for_empty(x)
            return []

        edge_out = self.edges[node_str][weights.argmax().item()](x_past=x)
        self.intermediate_outputs[node_str] = edge_out
        return edge_out[0]

    """
    # forward with a specific structure
    def forward_dynamic(self, s0: torch.Tensor, s1: torch.Tensor, structure):
        states = self.input_preprocessing(s0, s1)

        for i in range(2, self.max_nodes):
            cur_op_node = structure.nodes[i - 2]
            s_cur = 0.
            for op_name, j in cur_op_node:
                node_str = "{:}<-{:}".format(i, j)
                if node_str in self.edges:
                    edge = self.edges[node_str]
                    op_index = edge.index(op_name)
                    s_cur += edge[op_index](states[j])
            states.append(s_cur)
        return self.process_output(states)
    """


class SearchGDASDecoderCell(SearchGDASEncoderCell):
    op_types = MixedDecoderOps

    def get_edge_out_gdas(self, node_str, x, w_dag, index, cell_encoder_output, net_encoder_output, **kwargs):
        weights = w_dag[self.edge2index[node_str]]
        argmaxs = index[self.edge2index[node_str]].item()
        encoder_out_edge = cell_encoder_output[node_str]

        edge = self.edges[node_str]
        edge_out = edge[argmaxs](x_future=x,
                                 encoder_output_net=net_encoder_output,
                                 encoder_output_layer=encoder_out_edge[0],
                                 hx1=encoder_out_edge[1],
                                 hx2=encoder_out_edge[2], )
        weigsum = sum(
            weights[_ie] * edge_out if _ie == argmaxs else weights[_ie] for _ie in range(self.n_ops)
        )
        return weigsum

    def get_edge_out_gdasv1(self, node_str, x, w_dag, index, cell_encoder_output, net_encoder_output, **kwargs):
        weights = w_dag[self.edge2index[node_str]]

        if torch.all(weights <= 0.0):
            self.intermediate_outputs[node_str] = self.get_output_for_empty(x)
            return []

        argmaxs = index[self.edge2index[node_str]].item()
        encoder_out_edge = cell_encoder_output[node_str]

        edge_out = self.edges[node_str][argmaxs](x_future=x,
                                                 encoder_output_net=net_encoder_output,
                                                 encoder_output_layer=encoder_out_edge[0],
                                                 hx1=encoder_out_edge[1],
                                                 hx2=encoder_out_edge[2], )
        edge_out = weights[argmaxs] * edge_out
        return edge_out[0]

    def get_edge_out_select(self, node_str, x, w_dag, index, cell_encoder_output, net_encoder_output, **kwargs):
        weights = w_dag[self.edge2index[node_str]]
        encoder_out_edge = cell_encoder_output[node_str]

        if torch.all(weights <= 0.0):
            self.intermediate_outputs[node_str] = self.get_output_for_empty(x)
            return []

        edge_out = self.edges[node_str][weights.argmax().item()](x_future=x,
                                                                 encoder_output_net=net_encoder_output,
                                                                 encoder_output_layer=encoder_out_edge[0],
                                                                 hx1=encoder_out_edge[1],
                                                                 hx2=encoder_out_edge[2], )
        self.intermediate_outputs[node_str] = edge_out
        return edge_out[0]


class SearchDARTSFlatEncoderCell(SearchDARTSEncoderCell):
    op_types = MixedFlatEncoderOps

    def __init__(self,
                 n_nodes: int,
                 window_size: int,
                 forecasting_horizon: int,
                 n_input_nodes: int = 1,
                 PRIMITIVES: list[str] = PRIMITIVES_FLAT_ENCODER.keys(),
                 OPS_kwargs: dict[str, dict] = {},
                 is_last_cell: bool = False, ):
        nn.Module.__init__(self)
        self.max_nodes = n_nodes + n_input_nodes
        self.window_size = window_size
        self.forecasting_horizon = forecasting_horizon
        self.n_input_nodes = n_input_nodes
        self.PRIMITIVES = PRIMITIVES
        self.OPS_kwargs = OPS_kwargs
        self.is_last_cell = is_last_cell

        self.op_names_all = copy.deepcopy(PRIMITIVES)
        self.n_ops = len(self.op_names_all)


        self.edges = nn.ModuleDict()

        for i in range(n_input_nodes, self.max_nodes):
            # The first 2 nodes are input nodes
            for j in range(i):
                if is_last_cell:
                    if i == self.max_nodes - 1 and j == i - 1:
                        if 'mlp' in OPS_kwargs:
                            OPS_kwargs['mlp'].update(
                                {'is_last_layer': True}
                            )
                        else:
                            OPS_kwargs.update({'mlp': {'is_last_layer': True}})
                node_str = f"{i}<-{j}"
                op = self.op_types(window_size=window_size,
                                   forecasting_horizon=forecasting_horizon,
                                   PRIMITIVES=PRIMITIVES,
                                   OPS_kwargs=OPS_kwargs)  # TODO check if PRIMITIVES fits the requirements?
                self.edges[node_str] = op

        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def aggregate_edges_outputs(self, s_curs: list[torch.Tensor]):
        if len(s_curs) > 0:
            return sum(s_curs) / len(s_curs)
        return 0

    def get_edge_out(self, node_str, x, w_dag, alpha_prune_threshold, **kwargs):
        edge_out = self.edges[node_str](x_past=x, weights=w_dag[self.edge2index[node_str]],
                                        alpha_prune_threshold=alpha_prune_threshold)
        return edge_out


class SearchGDASFlatEncoderCell(SearchDARTSFlatEncoderCell, SearchGDASEncoderCell):
    op_types = MixedFlatEncoderOps

    def forward_gdas(self, s_previous: list[torch.Tensor], w_dag: list[list[float]], index, **kwargs):
        return SearchGDASEncoderCell.forward(self, s_previous=s_previous,
                                             w_dag=w_dag, index=index,
                                             edge_output_func=self.get_edge_out_gdas,
                                             **kwargs
                                             )

    def get_edge_out(self, node_str, x, w_dag, **kwargs):
        weights = w_dag[self.edge2index[node_str]]
        edge_out = self.edges[node_str](x_past=x, weights=weights)
        return edge_out

    def get_edge_out_gdas(self, node_str, x, w_dag, index, **kwargs):
        weights = w_dag[self.edge2index[node_str]]
        argmaxs = index[self.edge2index[node_str]].item()
        edge = self.edges[node_str]
        edge_out = edge[argmaxs](x_past=x)
        weigsum = sum(
            weights[_ie] for _ie in range(self.n_ops) if _ie != argmaxs
        )
        edge_out = weights[argmaxs] * edge_out + weigsum
        return edge_out

    def get_edge_out_gdasv1(self, node_str, x, w_dag, index, **kwargs):
        weights = w_dag[self.edge2index[node_str]]
        argmaxs = index[self.edge2index[node_str]].item()
        edge_out = self.edges[node_str][argmaxs](x_past=x)
        edge_out = weights[argmaxs] * edge_out
        return edge_out

    def get_edge_out_select(self, node_str, x, w_dag, **kwargs):
        weights = w_dag[self.edge2index[node_str]]
        edge_out = self.edges[node_str][weights.argmax().item()](x_past=x)
        return edge_out


class SampledEncoderCell(nn.Module):
    all_ops = PRIMITIVES_Encoder

    def __init__(self, n_nodes: int,
                 n_input_nodes: int,
                 d_model: int,
                 operations: list[int],
                 has_edges: list[bool],
                 PRIMITIVES: list[str] = PRIMITIVES_Encoder.keys(),
                 OPS_kwargs: dict[str, dict] = {},
                 is_first_cell: bool = False, ):
        """
        Args:
            n_nodes (int): Number of intermediate nodes. The output of the cell is calculated by
                concatenating the outputs of all intermediate nodes in the cell.
            d_model (int): model dimensions
            PRIMITIVES (list[str]): a list of all possible operations in this cells (the top one-shot model)
        """
        super().__init__()
        self.max_nodes = n_nodes + n_input_nodes
        self.n_input_nodes = n_input_nodes

        self.d_model = d_model

        # generate dag
        self.edges = nn.ModuleDict()

        k = 0
        for i in range(n_input_nodes, self.max_nodes):
            # The first 2 nodes are input nodes
            for j in range(i):
                if is_first_cell:
                    if i == n_input_nodes and j == 0:
                        if 'transformer' in OPS_kwargs:
                            OPS_kwargs['transformer'].update(
                                {'is_first_layer': True}
                            )
                        else:
                            OPS_kwargs.update({'transformer': {'is_first_layer': True}})
                dilation = int(2 ** (i - 1))
                if 'tcn' in OPS_kwargs:
                    OPS_kwargs['tcn'].update({'dilation': dilation})
                else:
                    OPS_kwargs.update({'tcn': {'dilation': dilation}})

                if has_edges[k]:
                    node_str = f"{i}<-{j}"
                    op_name = PRIMITIVES[operations[k]]
                    op_kwargs = OPS_kwargs.get(op_name, {})
                    op = self.all_ops[op_name](self.d_model, **op_kwargs)
                    self.edges[node_str] = op
                    k += 1

        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

        self.intermediate_outputs = {
            key: None for key in self.edge_keys
        }

    def forward(
            self,
            s_previous: list[torch.Tensor], **kwargs
    ):
        """Forward pass through the cell

        Args:
            s_previous: Output of the previous cells
            w_dag ( list[list[float]]): MixedOp weights ("alphas") (e.g. for n nodes and k primitive operations should
                be a list of length `n` of parameters where the n-th parameter has shape
                :math:`(n+2)xk = (number of inputs to the node) x (primitive operations)`)
                Each element in w_dag corresponds to an edge stored by self.edges
            alpha_prune_threshold:

        Returns:
            The output tensor of the cell
        """
        states = copy.copy(s_previous)

        for i in range(self.n_input_nodes, self.max_nodes):
            s_curs = []
            for j in range(i):
                node_str = f"{i}<-{j}"
                if node_str in self.edges:
                    s_curs.append(self.get_edge_out(node_str=node_str, x=states[j], **kwargs))

            states.append(self.aggregate_edges_outputs(s_curs))

        return self.process_output(states)

    def aggregate_edges_outputs(self, s_curs: list[torch.Tensor]):
        return sum(s_curs)

    def process_output(self, states: list[torch.Tensor]):
        return states[-1]

    def get_edge_out(self, node_str, x, **kwargs):
        edge_out = self.edges[node_str](x_past=x)
        self.intermediate_outputs[node_str] = edge_out
        return edge_out[0]


class SampledDecoderCell(SampledEncoderCell):
    all_ops = PRIMITIVES_Decoder

    def get_edge_out(self, node_str, x,
                     cell_encoder_output, net_encoder_output):
        encoder_out_edge = cell_encoder_output[node_str]
        edge_out = self.edges[node_str](x_future=x,
                                        encoder_output_net=net_encoder_output,
                                        encoder_output_layer=encoder_out_edge[0],
                                        hx1=encoder_out_edge[1],
                                        hx2=encoder_out_edge[2])

        return edge_out


class SampledFlatEncoderCell(SampledEncoderCell):
    all_ops = PRIMITIVES_FLAT_ENCODER

    def __init__(self, n_nodes: int,
                 n_input_nodes: int,
                 window_size: int,
                 forecasting_horizon: int,
                 operations: list[int],
                 has_edges: list[bool],
                 PRIMITIVES: list[str] = PRIMITIVES_Encoder.keys(),
                 OPS_kwargs: dict[str, dict] = {},
                 is_last_cell: bool = False, ):
        """
        Args:
            n_nodes (int): Number of intermediate nodes. The output of the cell is calculated by
                concatenating the outputs of all intermediate nodes in the cell.
            d_model (int): model dimensions
            PRIMITIVES (list[str]): a list of all possible operations in this cells (the top one-shot model)
        """
        nn.Module.__init__(self)
        self.max_nodes = n_nodes + n_input_nodes
        self.n_input_nodes = n_input_nodes

        self.window_size = window_size
        self.forecasting_horizon = forecasting_horizon

        # generate dag
        self.edges = nn.ModuleDict()

        k = 0
        for i in range(n_input_nodes, self.max_nodes):
            # The first 2 nodes are input nodes
            for j in range(i):
                if is_last_cell:
                    if i == self.max_nodes - 1 and j == i - 1:
                        if 'mlp' in OPS_kwargs:
                            OPS_kwargs['mlp'].update(
                                {'is_last_layer': True}
                            )
                        else:
                            OPS_kwargs.update({'transformer': {'is_last_layer': True}})
                if has_edges[k]:
                    node_str = f"{i}<-{j}"
                    op_name = PRIMITIVES[operations[k]]
                    op_kwargs = OPS_kwargs.get(op_name, {})
                    op = self.all_ops[op_name](window_size=self.window_size,
                                               forecasting_horizon=self.forecasting_horizon, **op_kwargs)
                    self.edges[node_str] = op
                    k += 1

        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

        self.intermediate_outputs = {
            key: None for key in self.edge_keys
        }

    def aggregate_edges_outputs(self, s_curs: list[torch.Tensor]):
        if len(s_curs) > 0:
            return sum(s_curs) / len(s_curs)
        return 0

    def get_edge_out(self, node_str, x, **kwargs):
        edge_out = self.edges[node_str](x_past=x)
        return edge_out
