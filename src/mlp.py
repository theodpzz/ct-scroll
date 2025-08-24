import torch
import torch.nn as nn

from argparse import Namespace

class MLP(nn.Module):
    def __init__(
        self, 
        args : Namespace = None,
    ) -> None:
        super(MLP, self).__init__()
        self.args = args
        embed_dim = args.embed_dim
        self.mlp  = GeGLU(embed_dim)

    def forward(
        self, 
        x,
    ) -> torch.Tensor:
        return self.mlp(x)
        

class GeGLU(nn.Module):
    def __init__(
        self,
        dim_in,
    ) -> None:
        super().__init__()
        self.gated_layers = nn.Linear(dim_in, dim_in*2, bias=False)
        self.act          = nn.GELU()

    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        h       = self.gated_layers(x)
        h, gate = h.chunk(2, dim=-1)
        h       = h * self.act(gate)
        return h
