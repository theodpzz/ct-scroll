<p align="center">
  <h2 align="center">[MIDL 2025] Imitating Radiological Scrolling: a Global-Local Attention Model for 3D Chest CT Volumes Multi-label Anomaly Classification 🩺👨🏻‍⚕️</h2>
</p>

✅ PyTorch [CT-Scroll](https://arxiv.org/pdf/2503.20652) model implementation.

## 🧩 Method Overview

The CT-Scroll architecture consists of three main components. (1) Axial slices of the volume are grouped into triplets and processed by a ResNet followed by a GAP layer, producing a vector representation per triplet. (2) The Scrolling Block then refines these embedded visual tokens using both global and local attention mechanisms. (3) Finally, the aggregated representations are fed into a classification head to predict anomalies.

<img src="https://github.com/theodpzz/ct-scroll/blob/master/figures/method_overview.png" alt="Method overview" width="900">

## Getting Started

### Clone the Repository

To clone this repository, use the following command:

```bash
git clone https://github.com/theodpzz/ct-scroll.git
```

### ⚡️ Installation

Make sure you have Python 3 installed. Then, install the dependencies using:

```bash
pip install -r requirements.txt
```

### 🚀 CT-Scroll Demo

To initialize the model with a classification head, use:

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

model = CT_Scroll(args)
```

To load [CT-Scroll weights pretrained on CT-RATE](https://huggingface.co/theodpzz/ct-scroll), use:

```python
ckpt = torch.load(path_ckpt)

model.load_state_dict(ckpt)
```

To perform a forward pass, use:

```python
volumes = torch.randn(4, 1, 240, 480, 480)
labels  = torch.randint(0, 2, (4, 18))

predictions, loss = model(volumes, labels)
```

### ⚙️ CT Scan Processing

CT scans are reformated such that the first axis points from Inferior to Superior, the second from Right to Left, and the third from Anterior to Posterior (SLP). The spacing (z, x, y) = (1.5, 0.75, 0.75) in millimeters. The Hounsfield Units are clipped to [-1000, +200], and mapped to the range [0, 1] before normalization using ImageNet statistic (-0.449).

<img src="https://github.com/theodpzz/ct-scroll/blob/master/figures/orientation.png" alt="Orientation" width="900">

## 🔗 Useful links

2D ResNet18 pretrained on ImageNet is available at: [https://download.pytorch.org/models/resnet18-f37072fd.pth](https://download.pytorch.org/models/resnet18-f37072fd.pth).

CT-Scroll pretrained weights on CT-RATE are available at [https://huggingface.co/theodpzz/ct-scroll](https://huggingface.co/theodpzz/ct-scroll).

## 🤝🏻 Acknowledgment

We thank contributors from the CT-RATE dataset available at [https://huggingface.co/datasets/ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE), and from the Rad-ChestCT dataset available at [https://zenodo.org/records/6406114](https://zenodo.org/records/6406114).

## 📎Citation

If you use this repository in your work, we would appreciate the following citation:

```bibtex
@InProceedings{dipiazza_2025_ctscroll,
        title = {Imitating Radiological Scrolling: A Global-Local Attention Model for 3D Chest CT Volumes Multi-Label Anomaly Classification},
        author = {Di Piazza, Theo and Lazarus, Carole and Nempont, Olivier and Boussel, Loic},
        booktitle = {Proceedings of The 8nd International Conference on Medical Imaging with Deep Learning -- MIDL 2025},
        year = {2025},
        publisher = {PMLR},
}
```
