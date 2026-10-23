"""Audit all 69 canonical LLFF/DTU results and rescore two saved PNGs per 3-view scene (CPU).

    bash scripts/run.sh tools scripts/verify_results.py --output docs/results_check.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fewshot.data import SCENES, load_scene, prior_dir  # noqa: E402

KEYS = ("PSNR", "SSIM", "LPIPS")


def check(dataset, views, name):
    setting = dataset if views == 3 else f"{dataset}_{views}v"
    out = ROOT / "experiments/dev/certisplat" / setting / name / "seed0"
    config = json.loads((out / "config.json").read_text())
    assert (config["dataset"], config["scene"], config["n_views"], config["seed"], config["iterations"]) == \
           (dataset, name, views, 0, 10000), out
    init = config["init"]
    assert init.startswith("npz:"), out
    path = Path(init[4:])
    if not path.is_absolute():
        path = ROOT / path
    assert path.resolve() == (prior_dir(dataset, name, views, "fused_np") / "init_points.npz").resolve(), out
    assert path.is_file() and (out / "point_cloud.ply").stat().st_size > 0, out
    log = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert log[-1]["step"] == 10000, out
    metrics = json.loads((out / "metrics.json").read_text())
    vertices = PlyData.read(out / "point_cloud.ply")["vertex"].data
    assert len(vertices) == metrics["points"] > 0, out
    expected = {"x", "y", "z", "opacity", *(f"f_dc_{k}" for k in range(3)),
                *(f"f_rest_{k}" for k in range(3 * ((config["sh_degree"] + 1) ** 2 - 1))),
                *(f"scale_{k}" for k in range(3)), *(f"rot_{k}" for k in range(4))}
    assert expected.issubset(vertices.dtype.names), out
    assert all(np.isfinite(vertices[key]).all() for key in vertices.dtype.names), out
    per_view = json.loads((out / "per_view.json").read_text())
    scene = load_scene(dataset, name, views)
    assert [row["view"] for row in per_view] == [view.name for view in scene.test], out
    for key in KEYS:
        values = [row[key] for row in per_view]
        assert np.isfinite(values).all(), (out, key)
        assert abs(float(np.mean(values)) - metrics[key]) < 1e-7, (out, key)
    renders = out / "test/renders"
    assert {p.stem for p in renders.glob("*.png")} == {view.name for view in scene.test}, out
    samples = []
    if views == 3:
        assert Path(config["output"]).resolve() == out.resolve(), out
        for index in (0, len(scene.test) - 1):
            view = scene.test[index]
            with Image.open(renders / f"{view.name}.png") as image:
                pred = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
            gt = np.floor(np.clip(view.load(), 0, 1) * 255 + 0.5) / 255
            mask = scene.test_mask(index)
            error = (pred - gt) ** 2
            mse = error.mean() if mask is None else error[mask == 1].mean()
            psnr = float(-10 * np.log10(mse))
            difference = abs(psnr - per_view[index]["PSNR"])
            assert difference < 5e-5, (out, view.name, difference)
            samples.append({"view": view.name, "PSNR": psnr, "saved_PSNR": per_view[index]["PSNR"],
                            "absolute_difference": difference})
    return {"dataset": dataset, "views": views, "scene": name, "prior": "fused_np", "seed": 0,
            "iterations": 10000, "test_views": len(per_view), "checkpoint_points": len(vertices),
            "checkpoint_parameters_finite": True, "png_samples": samples, "ok": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [check(dataset, views, name) for dataset in ("LLFF", "DTU") for views in (3, 6, 9)
            for name in SCENES[dataset]]
    errors = [sample["absolute_difference"] for row in rows for sample in row["png_samples"]]
    result = {"runs_checked": len(rows), "png_samples": len(errors), "max_PSNR_difference": max(errors),
              "settings": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(rows)} runs checked; {len(errors)} PNG samples; max PSNR difference {max(errors):.8f} dB")


if __name__ == "__main__":
    main()
