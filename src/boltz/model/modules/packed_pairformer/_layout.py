"""Ragged token/pair layout metadata."""

import torch


def _lengths(mask):
    # Preserve holes in a mask; only discard trailing padding.
    indices = torch.arange(mask.shape[-1], device=mask.device) + 1
    return tuple((mask.bool() * indices).amax(-1).clamp_min(1).cpu().tolist())


class Layout:
    """Offsets for token rows and flattened per-record pair squares."""

    def __init__(self, lengths, width, device):
        self.lengths = tuple(int(n) for n in lengths)
        self.width = width
        self.batch_size = len(lengths)
        if not lengths or min(lengths) < 1 or max(lengths) > width:
            raise ValueError("Invalid packed record lengths")
        self.offsets = [0]
        self.pair_offsets = [0]
        rows, pairs = [], []
        for i, n in enumerate(lengths):
            self.offsets.append(self.offsets[-1] + n)
            self.pair_offsets.append(self.pair_offsets[-1] + n * n)
            r = torch.arange(n, device=device)
            rows.append(i * width + r)
            pairs.append((i * width * width + r[:, None] * width + r).flatten())
        self.row_indices = torch.cat(rows)
        self.pair_indices = torch.cat(pairs)
        self.offsets_gpu = torch.tensor(self.offsets, device=device, dtype=torch.int32)
        self.pair_offsets_gpu = torch.tensor(
            self.pair_offsets, device=device, dtype=torch.int32
        )
        self.lengths_gpu = torch.tensor(lengths, device=device, dtype=torch.int32)
        self.is_dense = all(n == width for n in lengths)
        self.kernel_tasks = {}

    def pack(self, x, pair=False):
        rank = 3 if pair else 2
        flat = x.reshape(-1, *x.shape[rank:])
        if self.is_dense:
            return flat
        return flat.index_select(0, self.pair_indices if pair else self.row_indices)

    def unpack(self, x, pair=False):
        shape = (self.batch_size, self.width)
        if pair:
            shape += (self.width,)
        shape += x.shape[1:]
        if self.is_dense:
            return x.reshape(shape)
        out = x.new_zeros(shape)
        out.reshape(-1, *x.shape[1:]).index_copy_(
            0, self.pair_indices if pair else self.row_indices, x
        )
        return out

    def parts(self, x, pair=False):
        offsets = self.pair_offsets if pair else self.offsets
        for i, n in enumerate(self.lengths):
            shape = (1, n, n) if pair else (1, n)
            yield x[offsets[i] : offsets[i + 1]].reshape(*shape, *x.shape[1:])
