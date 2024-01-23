import torch
from torch import nn

from tsf_oneshot.cells.encoders.components import _Chomp1d
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import (
    PositionalEncoding
)


class EmbeddingLayer(nn.Module):
    # https://github.com/cure-lab/LTSF-Linear/blob/main/layers/Embed.py
    def __init__(self, c_in, d_model, kernel_size=2, dilation:int=1):
        super(EmbeddingLayer, self).__init__()
        padding = (kernel_size - 1) * dilation
        #"""
        self.tokenConv = nn.Conv1d(in_channels=c_in,
                                   out_channels=d_model,
                                   dilation=dilation,
                                   kernel_size=kernel_size, padding=padding,
                                   bias=False
                                   )
        self.chomp1 = _Chomp1d(padding)
        #"""
        self.c_in = c_in
        self.d_model = d_model
        #self.embedding = nn.Linear(
        #    c_in, d_model, bias=False
        #)

    def forward(self, x_past: torch.Tensor):
        return self.chomp1(self.tokenConv(x_past.permute(0, 2, 1))).transpose(1, 2)
        #return self.embedding(x_past)
