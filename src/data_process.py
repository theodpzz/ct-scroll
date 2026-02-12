import torch
import numpy as np

def process_volume(
    path,
    dd = 240,
    dh = 480,
    dw = 480,
) -> torch.tensor:

    # Warning: To adjust 
    # Assuming that the CT scan is already formatted with SLP orientation
    array = np.load(path)['arr_0']

    # Array to tensor
    tensor = torch.tensor(array)

    # Clip Hounsfield Units to [-1000, +200]
    tensor = torch.clip(tensor, -1000., +200.)

    # Shift to [0, +1200]
    tensor = tensor + torch.tensor(+1000., dtype=torch.float32)

    # Map [0, +1200] to [0, 1]
    tensor = tensor / torch.tensor(+1200., dtype=torch.float32)

    # ImageNet Normalization
    tensor = tensor + torch.tensor(-0.449, dtype=torch.float32)

    # CT scan dimensions
    d, h, w = tensor.shape

    # Crop
    h_start = max((h - dh) // 2, 0)
    h_end   = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end   = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end   = min(d_start + dd, d)

    tensor = tensor[d_start:d_end, h_start:h_end, w_start:w_end]

    # Pad
    pad_h_before = (dh - tensor.size(1)) // 2
    pad_h_after  = dh - tensor.size(1) - pad_h_before
    pad_w_before = (dw - tensor.size(2)) // 2
    pad_w_after  = dw - tensor.size(2) - pad_w_before
    pad_d_before = (dd - tensor.size(0)) // 2
    pad_d_after  = dd - tensor.size(0) - pad_d_before

    tensor = torch.nn.functional.pad(
        input = tensor, 
        pad   = (pad_w_before, pad_w_after, pad_h_before, pad_h_after, pad_d_before, pad_d_after), 
        value = -0.449
    )

    # Unsqueeze
    tensor = tensor.unsqueeze(0)

    return tensor
