import torch
from torch import nn

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import (
    PositionalEncoding
)

# _Chomp1d, _TemporalBlock and _TemporalConvNet original implemented by
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py, Carnegie Mellon University Locus Labs
# Paper: https://arxiv.org/pdf/1803.01271.pdf
class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super(_Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


def fold_tensor(x:torch.Tensor, skip_size:int):
    B, L, N = x.shape
    L_add = L % skip_size
    if L_add != 0:
        x = nn.functional.pad(x, (0, 0, 0, skip_size - L_add), 'constant', 0)
    x = x.view(B, -1, skip_size, N).transpose(2, 1).reshape(B * skip_size, -1, N)
    return x, (B, L, N)


def unfold_tensor(x:torch.Tensor, skip_size:int, x_shape_info: tuple[torch.Tensor]):
    B, L, N = x_shape_info
    x = x.view(B, skip_size, -1, N).transpose(2, 1).reshape(B, -1, N)[:, :L, :]
    return x


def check_node_is_connected_to_out(node_idx, n_nodes_max:int, nodes_to_remove: set, edges):
    if node_idx in nodes_to_remove:
        for i in range(node_idx, n_nodes_max):
            edge = f'{i}<-{node_idx}'
            if edge in edges:
                if i == n_nodes_max - 1:
                    nodes_to_remove.remove(node_idx)
                    return True
                else:
                    is_connect_to_out = check_node_is_connected_to_out(i, n_nodes_max, nodes_to_remove, edges)
                    if is_connect_to_out:
                        nodes_to_remove.remove(node_idx)
                        return True
    return False


class EmbeddingLayer(nn.Module):
    def __init__(self, c_in, d_model):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Linear(
            c_in, d_model
        )

    def forward(self, x_past: torch.Tensor):
        return self.embedding(x_past)
