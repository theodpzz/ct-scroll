import torch
import torch.nn as nn

from argparse import Namespace

from src.mlp import MLP

class LowerBlock(nn.Module):
    def __init__(
            self, 
            args : Namespace = None, 
            window_size : list = [16], 
            embed_dim : int = 512, 
            num_heads : int = 8, 
            dropout : float = 0.1,
        ) -> None:
        super().__init__()
        self.args = args

        self.window_size = window_size

        self.norm_in = nn.LayerNorm(embed_dim)
        
        self.norm_out = nn.LayerNorm(embed_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP
        self.mlp = MLP(args=args)

    def get_lower_mask(
            self,
            bs : int = 1,
            num_heads : int = 8,
            seq_len : int = 80,
            window_size : int = 4,
        ) -> torch.Tensor:

        seq_len             = self.args.nb_triplets
        sliding_window_mask = torch.zeros(seq_len, seq_len)

        for i in range(seq_len):
            sliding_window_mask[i, max(0, i - window_size + 1):i + 1] = 1

        sliding_window_mask = sliding_window_mask.bool()
        attn_mask           = ~sliding_window_mask

        return attn_mask.repeat(bs*num_heads, 1, 1)

    def forward(
        self,
        x : torch.Tensor,
    ) -> torch.Tensor:
        
        # Shortcut for residual connexion
        shortcut = x
        
        # First Normalize features
        x = self.norm_in(x)
        
        # Multi-Head Attention
        bs        = x.size(0)
        attn_mask = self.get_lower_mask(bs, window_size=self.window_size)
        attn_mask = attn_mask.to(x.device)
        x, _      = self.attention(query=x, key=x, value=x, attn_mask=attn_mask)
        
        # Residual connexion with shortcut
        x = shortcut + x
        
        # Final normalization layer
        xn = self.norm_out(x)
        
        # MLP
        x = x + self.mlp(xn)
        
        return x

class UpperBlock(nn.Module):
    def __init__(
        self,
        args : Namespace = None,
        window_size : list = 16,
        embed_dim : int = 512,
        num_heads : int = 8,
        dropout : float = 0.1,
    ) -> None:
        super().__init__()
        self.args = args

        # Window size
        self.window_size = window_size

        # First Normalization Layer
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Last Normalization Layer
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True
        )
        
        # FFN for further processing
        self.mlp = MLP(args=args)

    def get_upper_mask(
        self,
        bs : int = 4,
        num_heads : int = 8,
        seq_len : int = 80,
        window_size : int = 16,
    ) -> torch.Tensor:
        seq_len                   = self.args.nb_triplets
        upper_sliding_window_mask = torch.zeros(seq_len, seq_len)

        for i in range(seq_len):
            upper_sliding_window_mask[i, i:min(seq_len, i + window_size)] = 1

        upper_sliding_window_mask = upper_sliding_window_mask.bool()
        upper_attn_mask           = ~upper_sliding_window_mask

        return upper_attn_mask.repeat(bs*num_heads, 1, 1)

    def forward(
        self,
        x : torch.Tensor
    ) -> None:
        
        # Shortcut for residual connexion
        shortcut = x
        
        # First Normalize features
        x = self.norm1(x)
        
        # Multi-Head Attention
        bs        = x.size(0)
        attn_mask = self.get_upper_mask(bs, window_size=self.window_size)
        attn_mask = attn_mask.to(x.device)
        x, _      = self.attention(query=x, key=x, value=x, attn_mask=attn_mask)
        
        # Residual connexion with shortcut
        x = shortcut + x
        
        # Final normalization layer
        xn = self.norm2(x)
        
        # MLP
        x = x + self.mlp(xn)
        
        return x
  
class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        args : Namespace = None,
        embed_dim : int = 512,
        num_heads : int = 8,
        dropout : float = 0.1,
    ) -> None:
        super().__init__()
        self.args = args

        self.norm_in = nn.LayerNorm(embed_dim)
        
        self.norm_out = nn.LayerNorm(embed_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True
        )
        
        self.mlp = MLP(args=args)

    def forward(
        self,
        x : torch.Tensor,
    ) -> torch.Tensor:
        
        # Shortcut for residual connexion
        shortcut = x
        
        # First Normalize features
        x = self.norm_in(x)
        
        # Multi-Head Attention
        x, _ = self.attention(query=x, key=x, value=x)
        
        # residual connexion with shortcut
        x = shortcut + x
        
        # Final normalization layer
        xn = self.norm_out(x)
        
        # MLP
        x = x + self.mlp(xn)
        
        return x

class AlternateAttention(nn.Module):
    def __init__(
        self, 
        args : Namespace = None,
        window_size : int = 16,
    ) -> None:
        super().__init__()

        # arguments
        self.args = args

        # Global Attention
        self.global_block = TransformerEncoderBlock(args)

        # Local Attention 1
        self.lower_block = LowerBlock(args, window_size)

        # Local Attention 2
        self.upper_block = UpperBlock(args, window_size)

    def forward(self, x):

        x = self.global_block(x)

        x = self.lower_block(x)

        x = self.upper_block(x)

        return x
    
