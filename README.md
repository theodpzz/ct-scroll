### CT-Scroll: A Global-Local Attention Model for 3D Chest CT Volumes 🩺👨🏻‍⚕️

PyTorch [CT-Scroll](https://arxiv.org/abs/2503.20652) model implementation.

## Method Overview

The CT-Scroll architecture consists of three main components. (1) Axial slices of the volume are grouped into triplets and processed by a ResNet followed by a GAP layer, producing a vector representation per triplet. (2) The Scrolling Block then refines these embedded visual tokens using both global and local attention mechanisms. (3) Finally, the aggregated representations are fed into a classification head to predict anomalies.

<img src="https://github.com/theodpzz/ct-scroll/blob/master/figures/method_overview.png" alt="Method overview" width="900">

## Getting Started

### Clone the Repository

To clone this repository, use the following command:

```bash
git clone https://github.com/theodpzz/ct-scroll.git
```

### Installation

Make sure you have Python 3 installed. Then, install the dependencies using:

```bash
pip install -r requirements.txt
```

### Demo

```python
import torch
from argparse import Namespace
from src.model import CT_Scroll

args             = Namespace()
args.n_outputs   = 18
args.embed_dim   = 512
args.depth       = 1
args.window_size = [16]
args.nb_triplets = 80
args.path_resnet = None

device = torch.device('cpu')

model = CT_Scroll(args)
model.to(device);

volumes = torch.randn(1, 1, 240, 480, 480).to(device)
labels  = torch.randint(0, 2, (1, 18)).to(device)

predictions, loss = model(volumes, labels)
```