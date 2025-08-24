import torch
import torch.nn as nn

from argparse import Namespace
from torchvision import models

from src.attention import AlternateAttention

class CT_Scroll(nn.Module):
    def __init__(
        self, 
        args : Namespace = None,
    ) -> None:
        super(CT_Scroll, self).__init__()  

        self.args = args
        n_outputs = args.n_outputs
        self.d    = args.embed_dim
        
        self.features = self.get_resnet(args.path_resnet)
        
        self.projection_gap = nn.Sequential(
            nn.Linear(args.embed_dim, args.embed_dim), 
            nn.ReLU(True), 
            nn.Dropout(0.2),
            )

        # Shift Window Attention
        self.blocks = nn.ModuleList()
        for i in range(args.depth):
            block = AlternateAttention(args=args, window_size=self.args.window_size[i])
            self.blocks.append(block)

        # normalization layer for slice features
        self.layer_norm_out = nn.LayerNorm(args.embed_dim)

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(args.embed_dim, 128), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, n_outputs))
        
        # loss function
        self.loss = nn.BCEWithLogitsLoss(reduction = 'none')

        # sigmoid to compute probabilities
        self.activation = nn.Sigmoid()

    def get_resnet(
            self, 
            path: str = None,
        ) -> nn.Module:
        resnet = models.resnet18(weights=None)
        if path is not None:
            resnet.load_state_dict(torch.load(self.args.path_resnet, weights_only=True))
        return nn.Sequential(*(list(resnet.children())[:-2]))

    def getloss(
        self, 
        prediction: torch.Tensor, 
        target: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.loss(prediction, target.float())
        return loss

    def forward(
        self, 
        volume: torch.Tensor, 
        labels: torch.Tensor,
    ) -> torch.Tensor:

        # Extract batch size
        batch_size = volume.size(0)

        # Reshape by grouping channels 3 by 3
        x = volume.reshape(batch_size*80, 3, 480, 480) # [batch_size*N, 3, H, W]

        # Extract embeddings with resnet
        x = self.features(x)                           # [batch_size*N, d, h, w]

        # Global Average Pooling from ResNet feature maps
        x_gap = x.mean(dim=(2, 3))                     # [batch_size*N, d]

        # Reshape features with correct batch size
        x_gap = x_gap.view(batch_size, 80, self.d)     # [batch_size, N, d]
        
        # Project local embeddings
        x_gap = self.projection_gap(x_gap)             # [batch_size, N, d]

        # Global transformer
        for blk in self.blocks:
            x_gap = blk(x_gap)                         # [batch_size, N, d]

        # Normalize slices branch features if needed
        x_gap = self.layer_norm_out(x_gap)             # [batch_size, N, d]

        # Pooling
        x_gap = torch.mean(x_gap, dim=1)               # [batch_size, d]
        
        # Classifier
        logits = self.classifier(x_gap)                # [batch_size, M]
        
        # Loss
        loss = self.getloss(logits, labels)            # [batch_size, 1]

        # Activation
        predictions = self.activation(logits)          # [batch_size, M]

        return predictions, loss
