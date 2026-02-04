<p align="center">
  <h2 align="center">[MIDL 2025] Imitating Radiological Scrolling: a Global-Local Attention Model for 3D Chest CT Volumes Multi-label Anomaly Classification 🩺👨🏻‍⚕️</h2>
</p>

✅ PyTorch [CT-Scroll](https://arxiv.org/pdf/2503.20652) model implementation.

## 🧩 Method Overview

The CT-Scroll architecture consists of three main components. (1) Axial slices of the volume are grouped into triplets and processed by a ResNet followed by a GAP layer, producing a vector representation per triplet. (2) The Scrolling Block then refines these embedded visual tokens using both global and local attention mechanisms. (3) Finally, the aggregated representations are fed into a classification head to predict anomalies.

<img src="https://github.com/theodpzz/ct-scroll/blob/master/figures/method_overview.png" alt="Method overview" width="900">

## 🚀 Getting Started

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

### CT Scan Processing

CT scans are reformated such that the first axis points from Inferior to Superior, the second from Right to Left, and the third from Anterior to Posterior (SLP). The spacing (z, x, y) = (1.5, 0.75, 0.75) in millimeters. The Hounsfield Unit range [-1000, +200] is clipped to the range [0, 1] and normalized using ImageNet statistic (-0.449).

<img src="https://github.com/theodpzz/ct-ssg/blob/main/figures/orientation.png" alt="Orientation" width=


### Useful links

2D ResNet18 pretrained on ImageNet is available at: [https://download.pytorch.org/models/resnet18-f37072fd.pth](https://download.pytorch.org/models/resnet18-f37072fd.pth).
