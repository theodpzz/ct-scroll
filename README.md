<p align="center">
  <h2 align="center">Imitating Radiological Scrolling: a Global-Local Attention Model for 3D Chest CT Volumes Multi-label Anomaly Classification 🩺👨🏻‍⚕️</h2>
  <h4 align="center"><b>MIDL 2025</b></h4>
  <p align="center">
    <a href="https://arxiv.org/pdf/2503.20652"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2503.20652-b31b1b.svg"></a>
  </p>
</p>

---

## 🧩 Method Overview

The CT-Scroll architecture consists of three main components. (1) Axial slices of the volume are grouped into triplets and processed by a ResNet followed by a GAP layer, producing a vector representation per triplet. (2) The Scrolling Block then refines these embedded visual tokens using both global and local attention mechanisms. (3) Finally, the aggregated representations are fed into a classification head to predict anomalies.

<img src="https://github.com/theodpzz/ct-scroll/blob/main/figures/method_overview.png" alt="Method overview" width="900">

---

### Notice

This repository is currently under review for compliance with institutional and collaborative agreements.

The repository will be made publicly available once the approval process is completed.

---

## 🤝🏻 Acknowledgment

We thank contributors from the CT-RATE dataset available at [https://huggingface.co/datasets/ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE), and from the Rad-ChestCT dataset available at [https://zenodo.org/records/6406114](https://zenodo.org/records/6406114).

---

## Purpose

This code is provided for **academic and research purposes only**, to support reproducibility of the results described in the associated paper. This repository is a research prototype, and is not intended for clinical use.

---

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
