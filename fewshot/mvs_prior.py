"""Fixed-pose COLMAP PatchMatch using the benchmark training images.

Run: CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh tools -m fewshot.mvs_prior --dataset LLFF --all
"""
import argparse
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .data import PRIORS, SCENES, load_scene, prior_dir, read_colmap, read_points3d
from .prior_io import median_downsample, read_array, save_prior, write_json

COLMAP = os.environ.get("COLMAP", shutil.which("colmap") or "colmap")   # CUDA-enabled COLMAP 3.8


def run_scene(dataset, name, args):
    started = time.perf_counter()
    scene = load_scene(dataset, name, args.n_views)
    out = prior_dir(dataset, name, args.n_views, "mvs")
    out.mkdir(parents=True, exist_ok=True)
    work = PRIORS / "_work" / dataset / f"{args.n_views}v" / name
    work.mkdir(parents=True, exist_ok=False)
    images, known, sparse, dense = [work / s for s in ("images", "known", "sparse", "dense")]
    for directory in (images, known, sparse):
        directory.mkdir()
    for view in scene.train:
        (images / view.path.name).symlink_to(view.path.resolve())
    database = work / "database.db"
    num_sources = min(len(scene.train) - 1, 4)
    meta = {"dataset": dataset, "scene": name, "n_views": args.n_views, "source": "mvs",
            "num_patch_match_src_images": num_sources,
            "train_names": [v.name for v in scene.train], "max_image_size": args.max_image_size,
            "colmap": COLMAP, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "fixed_poses": True, "input_points": 0, "geom_consistency": True,
            "match_ratio": args.match_ratio, "guided_matching": args.guided_matching,
            "resampling": "median of positive finite samples binned by pixel centre",
            "commands": [], "timings_seconds": {}}

    def run(command, **options):
        argv = [COLMAP, command]
        for key, value in options.items():
            argv.extend(["--" + key, str(value)])
        tick = time.perf_counter()
        print(f"{dataset}/{name}: {command}", flush=True)
        with open(out / "colmap.log", "a") as log:
            log.write("\n" + " ".join(argv) + "\n")
            log.flush()
            subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)
        meta["commands"].append(argv)
        meta["timings_seconds"][command] = time.perf_counter() - tick

    run("feature_extractor", database_path=database, image_path=images, **{
        "ImageReader.camera_model": "PINHOLE", "ImageReader.single_camera_per_image": 1,
        "SiftExtraction.gpu_index": 0, "SiftExtraction.num_threads": 8})
    if (scene.source / "sparse/0").is_dir():
        originals = {Path(n).name: (K, pose, size)
                     for n, K, pose, size in read_colmap(scene.source / "sparse/0")}
    else:  # Blender: no COLMAP model; the benchmark cameras at the images' native size
        originals = {}
        for view in scene.train:
            with Image.open(view.path) as image:
                w, h = image.size
            K = view.K.astype(np.float64).copy()
            K[0] *= w / view.width
            K[1] *= h / view.height
            originals[view.path.name] = (K, view.w2c.astype(np.float64), (h, w))
    camera_lines, image_lines = [], []
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT image_id, name, camera_id FROM images ORDER BY image_id").fetchall()
        for image_id, filename, camera_id in rows:
            K, pose, (h, w) = originals[filename]
            params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype="<f8")
            connection.execute("UPDATE cameras SET model=1, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
                               (w, h, params.tobytes(), camera_id))
            camera_lines.append(f"{camera_id} PINHOLE {w} {h} " + " ".join(map(str, params)))
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
            image_lines.append(f"{image_id} " + " ".join(map(str, [*quat, *pose[:3, 3]]))
                               + f" {camera_id} {filename}\n\n")
    (known / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (known / "images.txt").write_text("".join(image_lines))
    (known / "points3D.txt").write_text("")
    run("exhaustive_matcher", database_path=database, **{
        "SiftMatching.gpu_index": 0, "SiftMatching.num_threads": 8,
        "SiftMatching.guided_matching": int(args.guided_matching),
        "SiftMatching.max_ratio": args.match_ratio})
    run("point_triangulator", database_path=database, image_path=images,
        input_path=known, output_path=sparse, **{
            "Mapper.fix_existing_images": 1, "Mapper.ba_refine_focal_length": 0,
            "Mapper.ba_refine_principal_point": 0, "Mapper.ba_refine_extra_params": 0,
            "Mapper.tri_ignore_two_view_tracks": 0, "Mapper.num_threads": 8})
    reconstructed = read_colmap(sparse)
    assert {n for n, *_ in reconstructed} == {v.path.name for v in scene.train}
    for filename, K, pose, _ in reconstructed:
        np.testing.assert_allclose(K, originals[filename][0], rtol=0, atol=1e-9)
        np.testing.assert_allclose(pose, originals[filename][1], rtol=0, atol=1e-9)
    meta["sparse_points"] = len(read_points3d(sparse)[0])
    if not meta["sparse_points"]:
        raise RuntimeError(f"No {args.n_views}-training-view sparse points: {dataset}/{name}")
    run("image_undistorter", image_path=images, input_path=sparse, output_path=dense,
        output_type="COLMAP", max_image_size=args.max_image_size,
        num_patch_match_src_images=num_sources)
    if len(scene.train) == 3:
        # Preserve the original three-view source order exactly. For larger
        # splits COLMAP selects up to four sources using sparse-track overlap.
        filenames = [v.path.name for v in scene.train]
        (dense / "stereo/patch-match.cfg").write_text("".join(
            filename + "\n" + ", ".join(n for n in filenames if n != filename) + "\n"
            for filename in filenames))
    run("patch_match_stereo", workspace_path=dense, workspace_format="COLMAP", **{
        "PatchMatchStereo.gpu_index": 0, "PatchMatchStereo.geom_consistency": "true",
        "PatchMatchStereo.max_image_size": args.max_image_size})
    meta["views"] = {}
    undistorted = {n: (K, size) for n, K, _, size in read_colmap(dense / "sparse")}
    for view in scene.train:
        K, (h, w) = undistorted[view.path.name]
        scaled = K.copy()
        scaled[0] *= view.width / w
        scaled[1] *= view.height / h
        np.testing.assert_allclose(scaled, view.K, rtol=1e-6, atol=1e-5)
        raw = read_array(dense / "stereo/depth_maps" / (view.path.name + ".geometric.bin"))
        depth = median_downsample(raw, view.height, view.width)
        valid = depth > 0
        save_prior(out, view.name, depth, valid, valid)
        meta["views"][view.name] = {"coverage": float(valid.mean()), "raw_size": list(raw.shape)}
    shutil.rmtree(images)
    shutil.rmtree(dense / "images")
    shutil.rmtree(dense / "stereo/normal_maps")
    for path in (dense / "stereo/depth_maps").glob("*.photometric.bin"):
        path.unlink()
    meta["timings_seconds"]["total"] = time.perf_counter() - started
    write_json(out / "meta.json", meta)
    print(f"{dataset}/{name}: complete in {meta['timings_seconds']['total']:.1f}s; "
          f"coverage {[round(s['coverage'], 3) for s in meta['views'].values()]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SCENES, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--n-views", type=int, default=3)
    parser.add_argument("--max-image-size", type=int, default=2016)
    parser.add_argument("--guided-matching", action="store_true")
    parser.add_argument("--match-ratio", type=float, default=0.8)
    args = parser.parse_args()
    for name in SCENES[args.dataset] if args.all else [args.scene]:
        run_scene(args.dataset, name, args)


if __name__ == "__main__":
    main()
