"""Array I/O and pixel-centred resampling shared by depth-prior tools."""
import json

import numpy as np


def read_array(path):
    """COLMAP dense array: w&h&c& followed by column-major float32."""
    with open(path, "rb") as stream:
        header = bytearray()
        while header.count(b"&") < 3:
            char = stream.read(1)
            if not char:
                raise ValueError(f"Incomplete COLMAP header: {path}")
            header.extend(char)
        w, h, c = map(int, header[:-1].split(b"&"))
        array = np.fromfile(stream, dtype="<f4").reshape((w, h, c), order="F")
    return array.transpose(1, 0, 2).squeeze(axis=2) if c == 1 else array.transpose(1, 0, 2)


def nearest(array, height, width):
    """Nearest source centre to each target centre, without a half-pixel shift."""
    y = np.floor((np.arange(height) + 0.5) * array.shape[0] / height).astype(int)
    x = np.floor((np.arange(width) + 0.5) * array.shape[1] / width).astype(int)
    return array[y[:, None], x[None, :]]


def median_downsample(depth, height, width):
    """Median positive finite source samples whose centres fall in each target pixel."""
    h, w = depth.shape
    y, x = np.nonzero(np.isfinite(depth) & (depth > 0))
    bins = np.floor((y + 0.5) * height / h).astype(int) * width
    bins += np.floor((x + 0.5) * width / w).astype(int)
    values = depth[y, x]
    order = np.lexsort((values, bins))
    counts = np.bincount(bins, minlength=height * width)
    starts = np.cumsum(counts) - counts
    used = counts > 0
    mid1 = starts[used] + (counts[used] - 1) // 2
    mid2 = starts[used] + counts[used] // 2
    out = np.zeros(height * width, np.float32)
    out[used] = (values[order[mid1]] + values[order[mid2]]) * 0.5
    return out.reshape(height, width)


def save_prior(directory, name, depth, valid, conf):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"{name}.npz",
                        depth=np.where(valid, depth, 0).astype(np.float32),
                        valid=valid.astype(bool),
                        conf=np.where(valid, conf, 0).astype(np.float32))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
