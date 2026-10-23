"""Benchmark scenes: LLFF, DTU, Mip-NeRF360 and Blender with the sparse-view splits used by the local baselines.

Splits, image sizes and resampling follow StereoGS (the strongest local baseline):
LLFF and Mip-NeRF360 hold out every 8th image and take n training views by linspace over the rest;
DTU uses training ids [25, 22, 28, ...] and the RegNeRF test list; Blender uses the 8 training frames
[2, 16, 26, 55, 73, 76, 86, 93] and every 8th test frame, composited onto white. Images are read from
full resolution and resized with PIL's default (bicubic) filter.

Intrinsics use the COLMAP convention: the top-left image corner is (0, 0), so the
centre of pixel (i, j) is (i + 0.5, j + 0.5).
"""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct

import numpy as np
from PIL import Image

# Dataset root: $CERTISPLAT_DATA, else <repo>/datasets (see README).
DATA_ROOT = Path(os.environ.get("CERTISPLAT_DATA", Path(__file__).resolve().parents[1] / "datasets"))
SOURCES = {"LLFF": DATA_ROOT / "nerf_llff_data", "DTU": DATA_ROOT / "DTU/dtu_corgs",
           "Mip-NeRF360": DATA_ROOT / "mipnerf360", "Blender": DATA_ROOT / "nerf_synthetic"}
FACTORS = {"LLFF": 8, "DTU": 4, "Mip-NeRF360": 8, "Blender": 2}
WHITE_BACKGROUND = {"Blender"}      # RGBA images composited onto white, as the baselines train and evaluate
SCENES = {
    "LLFF": ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"],
    "DTU": ["scan8", "scan21", "scan30", "scan31", "scan34", "scan38", "scan40", "scan41",
            "scan45", "scan55", "scan63", "scan82", "scan103", "scan110", "scan114"],
    "Mip-NeRF360": ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"],
    "Blender": ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"],
}
BLENDER_TRAIN = [2, 16, 26, 55, 73, 76, 86, 93]   # StereoGS / DietNeRF 8-view frames of transforms_train
DTU_TRAIN = [25, 22, 28, 40, 44, 48, 0, 8, 13]   # standard order; n views use the first n
DTU_TEST = [1, 2, 9, 10, 11, 12, 14, 15, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35,
            41, 42, 43, 45, 46, 47]
DTU_MASKS = DATA_ROOT / "DTU/submission_data/idrmasks"
PRIORS = Path(__file__).resolve().parents[1] / "experiments/priors"


def prior_dir(dataset, scene, n_views, source=None):
    """experiments/priors/<dataset>/<n>v/<scene>[/<source>]: per-view-count prior storage."""
    path = PRIORS / dataset / f"{n_views}v" / scene
    return path / source if source else path
DTU_NESTED_MASKS = {"scan110", "scan114", "scan40", "scan55", "scan63"}


@dataclass
class View:
    name: str
    path: Path
    K: np.ndarray        # 3x3, at the benchmark resolution
    w2c: np.ndarray      # 4x4 world-to-camera (OpenCV axes)
    height: int
    width: int
    index: int           # position in the name-sorted image list (Blender: train frames, then test frames)
    white_background: bool = False

    def load(self):
        with Image.open(self.path) as image:
            if self.white_background and image.mode == "RGBA":
                # StereoGS loadCam: resize the RGBA image, then composite onto white in float.
                rgba = np.asarray(image.resize((self.width, self.height)), dtype=np.float32) / 255.0
                return rgba[..., :3] * rgba[..., 3:] + (1.0 - rgba[..., 3:])
            image = image.convert("RGB").resize((self.width, self.height))
            return np.asarray(image, dtype=np.float32) / 255.0


def _read(stream, fmt):
    return struct.unpack("<" + fmt, stream.read(struct.calcsize("<" + fmt)))


def _quat_to_rotation(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def read_colmap(sparse_dir):
    """Return name-sorted (name, K, w2c, (H, W)) from a COLMAP binary model."""
    cameras = {}
    with open(sparse_dir / "cameras.bin", "rb") as stream:
        for _ in range(_read(stream, "Q")[0]):
            camera_id, model, width, height = _read(stream, "iiQQ")
            params = _read(stream, "d" * {0: 3, 1: 4, 2: 4, 3: 5}[model])
            fx, fy, cx, cy = params[:4] if model == 1 else (params[0], params[0], params[1], params[2])
            cameras[camera_id] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), (height, width))
    images = []
    with open(sparse_dir / "images.bin", "rb") as stream:
        for _ in range(_read(stream, "Q")[0]):
            row = _read(stream, "idddddddi")
            name = bytearray()
            while (char := stream.read(1)) != b"\0":
                name.extend(char)
            stream.seek(24 * _read(stream, "Q")[0], 1)
            w2c = np.eye(4)
            w2c[:3, :3] = _quat_to_rotation(row[1:5])
            w2c[:3, 3] = row[5:8]
            K, size = cameras[row[8]]
            images.append((name.decode(), K.copy(), w2c, size))
    return sorted(images, key=lambda item: Path(item[0]).stem)


def read_points3d(sparse_dir):
    """COLMAP points3D.bin -> (xyz, rgb in [0, 1])."""
    xyz, rgb = [], []
    with open(sparse_dir / "points3D.bin", "rb") as stream:
        for _ in range(_read(stream, "Q")[0]):
            row = _read(stream, "QdddBBBd")
            xyz.append(row[1:4])
            rgb.append(row[4:7])
            stream.seek(8 * _read(stream, "Q")[0], 1)
    return np.array(xyz, np.float32), np.array(rgb, np.float32) / 255.0


@dataclass
class Scene:
    dataset: str
    name: str
    source: Path
    views: list          # all views, name-sorted
    train: list
    test: list

    @property
    def extent(self):
        """3DGS 'cameras_extent': 1.1 x radius of the training camera centres."""
        centers = np.stack([np.linalg.inv(v.w2c)[:3, 3] for v in self.train])
        return float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)

    def test_mask(self, position):
        """DTU evaluation mask (H, W, 3) in [0, 1] for the given test position; None for LLFF."""
        if self.dataset != "DTU":
            return None
        directory = DTU_MASKS / self.name
        if self.name in DTU_NESTED_MASKS:
            directory = directory / "mask"
        view = self.test[position]
        with Image.open(directory / f"{DTU_TEST[position]:03d}.png") as image:
            image = image.resize((view.width, view.height))
            return np.asarray(image, dtype=np.float32)[..., :3] / 255.0


def blender_views(source, split, factor):
    """Frames of transforms_<split>.json in file order (NeRF camera-to-world, OpenGL axes)."""
    meta = json.loads((source / f"transforms_{split}.json").read_text())
    views = []
    for index, frame in enumerate(meta["frames"]):
        path = source / (frame["file_path"] + ".png")
        with Image.open(path) as image:
            width, height = image.size
        focal = 0.5 * width / math.tan(0.5 * meta["camera_angle_x"])
        w, h = round(width / factor), round(height / factor)
        K = np.array([[focal * w / width, 0, 0.5 * w], [0, focal * h / height, 0.5 * h], [0, 0, 1]])
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        c2w[:3, 1:3] *= -1                      # OpenGL (y up, z back) -> OpenCV (y down, z forward)
        views.append(View(f"{split}_{path.stem}", path, K.astype(np.float32),
                          np.linalg.inv(c2w).astype(np.float32), h, w, index, white_background=True))
    return views


def load_scene(dataset, name, n_views=3, dense=False):
    """Benchmark split. dense=True trains on every non-test image (an upper-bound protocol)."""
    source = SOURCES[dataset] / name
    factor = FACTORS[dataset]
    if dataset == "Blender":
        train_all = blender_views(source, "train", factor)
        test = blender_views(source, "test", factor)[::8]
        train = train_all if dense else [train_all[i] for i in BLENDER_TRAIN[:n_views]]
        return Scene(dataset, name, source, train_all + test, train, test)
    views = []
    for index, (filename, K, w2c, (height, width)) in enumerate(read_colmap(source / "sparse/0")):
        h, w = round(height / factor), round(width / factor)
        K = K.copy()
        K[0] *= w / width
        K[1] *= h / height
        views.append(View(Path(filename).stem, source / "images" / Path(filename).name,
                          K.astype(np.float32), w2c.astype(np.float32), h, w, index))
    if dataset == "DTU":
        test = [views[i] for i in DTU_TEST]
        pool = [v for i, v in enumerate(views) if i not in DTU_TEST]
        train = pool if dense else [views[i] for i in DTU_TRAIN[:n_views]]
    else:
        test = views[::8]
        pool = [v for i, v in enumerate(views) if i % 8 != 0]
        picks = {round(i) for i in np.linspace(0, len(pool) - 1, n_views)}
        train = pool if dense else [v for i, v in enumerate(pool) if i in picks]
    return Scene(dataset, name, source, views, train, test)


def dtu_background_mask(image, scene_name):
    """DNGaussian/StereoGS DTU heuristic: a pixel is background if it and the 49 pixels
    above it are all dark. image: (H, W, 3)."""
    threshold = 15 / 255 if scene_name == "scan110" else 30 / 255
    dark = image.max(-1) < threshold
    mask = dark.copy()
    for shift in range(1, 50):
        mask[shift:] &= dark[:-shift]
    return mask
