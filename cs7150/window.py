import torch

def sliding_window(dims=32, window=1, stride=1, hole=False):
    if isinstance(dims, int):
        dims = (dims, dims)
    if isinstance(dims, torch.Tensor):
        dims = dims.shape
    assert(len(dims) == 2)
    for y in range(0, dims[0], stride):
        for x in range(0, dims[1], stride):
            mask = torch.zeros(*dims)
            mask[y:y+window, x:x+window] = 1
            if hole:
                mask = 1 - mask
            yield mask
