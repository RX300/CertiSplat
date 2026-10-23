"""Run the agreed baseline matrix using the existing official programs."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks/first_pass"  # replaced by --bench
RUN = ROOT / "scripts/run.sh"
SELF = Path(__file__).resolve()
MASKS = Path(os.environ.get("CERTISPLAT_DATA", str(ROOT / "datasets"))) / "DTU/submission_data/idrmasks"
METRICS = ("PSNR", "SSIM", "LPIPS")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def folder(dataset, views):
    """Result sub-folder: the dataset name for the first-pass view counts, '<dataset>_<n>v' otherwise."""
    return dataset if views == {"LLFF": 3, "DTU": 3, "Mip-NeRF360": 24, "Blender": 8}[dataset] else f"{dataset}_{views}v"


def command(method, script, *args):
    return ["bash", str(RUN), method, str(script), *map(str, args)]


MATRICES = {
    # (dataset, views, evaluation resolution)
    "first_pass": [("LLFF", 3, 8), ("DTU", 3, 4), ("Mip-NeRF360", 24, 8), ("Blender", 8, 2)],
    "views_69": [("LLFF", 6, 8), ("LLFF", 9, 8), ("DTU", 6, 4), ("DTU", 9, 4)],
    "other_datasets": [("Mip-NeRF360", 12, 8), ("Mip-NeRF360", 24, 8), ("Blender", 8, 2)],
}
# Methods per matrix (StableGS dropped after the 3-view pass at the user's request); Blender is StereoGS only.
METHODS = {"first_pass": ["StableGS", "TWINGS", "StereoGS"], "views_69": ["StableGS", "TWINGS", "StereoGS"],
           "other_datasets": ["TWINGS", "StereoGS"]}


def plan(matrix):
    BENCH.mkdir(parents=True, exist_ok=True)
    if (BENCH / "manifest.json").exists():
        raise FileExistsError("Benchmark already planned; use run to resume.")
    deployment = read_json(ROOT / "deployment_status.json")
    jobs = []
    # Later matrices also free the GUI port (parallel jobs on one GPU); it is not a training setting.
    gui = [] if matrix == "first_pass" else ["--ip", "127.0.0.1", "--port", 0]
    for dataset, views, eval_resolution in MATRICES[matrix]:
        data = deployment["datasets"][dataset]
        for scene in data["scenes"]:
            methods = ["StereoGS"] if dataset == "Blender" else METHODS[matrix]
            for method in methods:
                source = Path(data["path"]) / scene
                output = BENCH / "models" / method / folder(dataset, views) / scene
                training_resolution = 4 if dataset == "Mip-NeRF360" and method != "StereoGS" else eval_resolution
                notes = []
                if training_resolution != eval_resolution:
                    notes.append("Official 1/4 training; common 1/8 evaluation, not native-resolution paper metrics")
                steps = []
                common = ["-s", source, "-m", output, "--eval", "-r", training_resolution, "--n_views", views]
                if method == "StableGS":
                    entry = "train3.py" if dataset == "LLFF" else "train5.py"
                    first_stage = 30000 if dataset == "Mip-NeRF360" else 10000
                    iteration = 5000
                    stage_iterations = [first_stage, 5000]
                    extra = []
                    if dataset == "DTU" and scene == "scan110" and views == 3:
                        extra = ["--rand_pcd"]
                        notes.append("No supplied 3-view COLMAP cloud; official --rand_pcd option, explicitly selected locally")
                    if dataset == "LLFF" and scene == "fern" and views == 3:
                        output = ROOT / "outputs/stablegs_llff_3v/fern"
                        notes.append("Reuse verified complete 10000+5000-step official run; exclude earlier smoke runs")
                    else:
                        steps.append({"name": "train", "command": command(method, entry, *common, *extra, *gui,
                            "--iterations", first_stage, "--test_iterations", 5000, first_stage,
                            "--save_iterations", 5000, first_stage)})
                    notes.append(f"Official {entry}; evaluate stage-two iteration_5000")
                elif method == "TWINGS":
                    iteration = 10000
                    stage_iterations = [iteration]
                    pcd = source / f"multiview_pcd/{views}_views/twings_init_pcd.ply"
                    init_dataset = {"LLFF": "nerf_llff_data", "DTU": "DTU", "Mip-NeRF360": "mipnerf360"}[dataset]
                    if not pcd.is_file():
                        steps.append({"name": "initialize", "command": command("tools", ROOT / "scripts/init_twings.py",
                            "--dataset", init_dataset, "--scene", scene, "--views", views)})
                    steps.append({"name": "train", "command": command(method, "train.py", *common, *gui,
                        "--pcd_path", pcd, "--depth_loss", "--depth_weight", 0.01 if dataset == "DTU" else 0.03,
                        "--depth_pseudo_weight", 0.4 if dataset == "Mip-NeRF360" else 0.1,
                        "--iterations", iteration, "--test_iterations", iteration, "--save_iterations", iteration)})
                else:
                    iteration = 7000 if dataset == "Blender" else 30000
                    stage_iterations = [iteration]
                    name = "MipNeRF360" if dataset == "Mip-NeRF360" else dataset
                    height, width = (800, 800) if dataset == "Blender" else (1440, 1920)
                    mvs = "MVSFormer" if dataset == "DTU" else "MVSAnywhere"
                    shift = {"LLFF": 4, "DTU": 0.5, "Mip-NeRF360": 0.6, "Blender": 0.6}[dataset]
                    extra = ["--white_background"] if dataset == "Blender" else []
                    steps.append({"name": "train", "command": command(method, "train.py", *common, *extra, *gui,
                        "--dataset_name", name, "--use_mvs_init", "--mvs_model_name", mvs,
                        "--resize_mvs_depth_H", height, "--resize_mvs_depth_W", width,
                        "--img_dist_thresh", 50 if dataset == "DTU" else 3, "--depth_thresh", 0.08, "--vthresh", 2,
                        "--init_pcd_downsample", 0.1, "--dropout_factor", 0.3, "--sigma_noise", 0,
                        "--opacity_decay", "--opacity_decay_factor", 0.99, "--grad_sensitivity", 0.5,
                        "--generate_binocular_view", "--start_generate_binocular_view_iters", 4000 if dataset == "Blender" else 20000,
                        "--cam_trans_dist", shift, "--binocular_model_type", "FoundationStereo", "--binocular_depth",
                        "--lambda_binocular", 1, "--iterations", iteration,
                        "--test_iterations", iteration, "--save_iterations", iteration)})
                    notes.append("README full-method configuration; fixed dropout=0.3 (StereoGS*)")
                    if dataset == "Blender":
                        notes.append("Enable the three full-method flags commented out in upstream Blender batch script, following README")
                render_args = ["-m", output, "-r", eval_resolution, "--iteration", iteration, "--skip_train"]
                if method == "StereoGS":
                    render_args += ["--dataset_name", "MipNeRF360" if dataset == "Mip-NeRF360" else dataset,
                                    "--n_views", views, "--dropout_factor", 0.3, "--eval"]
                    if dataset == "Blender" and matrix != "first_pass":
                        render_args += ["--white_background"]  # as the official Blender render command
                steps.append({"name": "render", "command": command(method, "render.py", *render_args)})
                steps.append({"name": "evaluate", "command": command("tools", SELF, "score", "--bench", BENCH.name,
                                                                      "--index", len(jobs))})
                jobs.append({"method": method, "dataset": dataset, "scene": scene, "views": views,
                    "train_resolution": training_resolution, "eval_resolution": eval_resolution,
                    "stage_iterations": stage_iterations, "iteration": iteration, "source": str(source),
                    "output": str(output), "notes": notes, "steps": steps})
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S %z"), "matrix": matrix, "jobs": jobs,
        "sources": {name: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT/name, text=True).strip()
                    for name in ["StableGS", "TWINGS", "StereoGS"]},
        "protocol": "One run per scene. Official training implementations; native training resolutions; common evaluation resolutions: LLFF 8, DTU 4, Mip-NeRF360 8, Blender 2. StereoGS official evaluator for all saved PNGs, LPIPS-vgg; public DTU foreground masks. Dataset scores are unweighted means over scenes. Partial coverage is never reported as a full-dataset mean."}
    write_json(BENCH / "manifest.json", manifest)
    write_json(BENCH / "status.json", {"status": "planned", "jobs": [
        {"status": "pending", "completed_steps": [], "seconds": 0} for _ in jobs]})
    collect()
    print(f"Planned {len(jobs)} scene runs at {BENCH}")


def job_dir(job):
    return BENCH / "logs" / job["method"] / folder(job["dataset"], job["views"]) / job["scene"]


def score(index):
    job = read_json(BENCH / "manifest.json")["jobs"][index]
    source = Path(job["output"]) / "test" / f"ours_{job['iteration']}"
    names = sorted(p.name for p in (source / "renders").glob("*.png"))
    if job["method"] == "StereoGS":
        names = [name for name in names if Path(name).stem.isdigit()]
    if job["dataset"] in ["DTU", "Blender"]:
        expected = 25
    else:
        image_count = len([p for p in (Path(job["source"]) / "images").iterdir()
                           if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        expected = (image_count + 7) // 8
    if len(names) != expected:
        raise ValueError(f"Expected {expected} test views, found {len(names)}: {source}")
    evaluation = BENCH / "evaluation" / job["method"] / folder(job["dataset"], job["views"]) / job["scene"]
    method_dir = evaluation / "test" / f"ours_{job['iteration']}"
    for kind in ["renders", "gt"]:
        directory = method_dir / kind
        directory.mkdir(parents=True, exist_ok=True)
        for number, name in enumerate(names):
            original = source / kind / name
            if not original.is_file():
                raise FileNotFoundError(original)
            link = directory / f"{number:05d}.png"
            if link.is_symlink():
                link.unlink()
            link.symlink_to(original)
    # This calls the unchanged official evaluator; only filenames are adapted.
    # Upstream catches evaluation errors, so require a fresh result file.
    (evaluation / "results.json").unlink(missing_ok=True)
    subprocess.run(command("StereoGS", "metrics.py", "-m", evaluation,
        "--dataset_name", job["dataset"], "--idrmasks_path", MASKS), cwd=ROOT, check=True)
    values = read_json(evaluation / "results.json")[f"ours_{job['iteration']}"]
    if not all(math.isfinite(values[key]) for key in METRICS):
        raise ValueError(f"Non-finite metric values: {values}")
    values["test_views"] = expected
    values["source"] = str(evaluation / "results.json")
    write_json(job_dir(job) / "scores.json", values)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect():
    manifest = read_json(BENCH / "manifest.json")
    state = read_json(BENCH / "status.json")
    scenes = []
    groups = {}
    for job, status in zip(manifest["jobs"], state["jobs"]):
        row = {key: job[key] for key in ["method", "dataset", "scene", "views", "train_resolution", "eval_resolution", "output"]}
        row.update(status=status["status"], seconds=round(status["seconds"], 2),
                   notes="; ".join(job["notes"]), stage_iterations="+".join(map(str, job["stage_iterations"])))
        if status["status"] == "completed":
            values = read_json(job_dir(job) / "scores.json")
            row.update({key: values[key] for key in METRICS})
        scenes.append(row)
        groups.setdefault((job["dataset"], job["views"], job["method"]), []).append(row)
    fields = ["method", "dataset", "scene", "views", "train_resolution", "eval_resolution", "stage_iterations",
              "status", *METRICS, "seconds", "notes", "output"]
    write_csv(BENCH / "per_scene.csv", scenes, fields)
    summaries = []
    for (dataset, _, method), rows in groups.items():
        complete = [r for r in rows if r["status"] == "completed"]
        summary = {"dataset": dataset, "method": method, "views": rows[0]["views"],
                   "train_resolution": rows[0]["train_resolution"], "eval_resolution": rows[0]["eval_resolution"],
                   "completed": len(complete), "expected": len(rows),
                   "failed": sum(r["status"] == "failed" for r in rows)}
        if all(r["status"] == "skipped" or r["status"] == "completed" for r in rows) and any(r["status"] == "skipped" for r in rows):
            summary["expected"] = f"{len(rows)} (skipped {sum(r['status'] == 'skipped' for r in rows)})"
        if complete:
            prefix = "" if len(complete) == len(rows) else "partial_"
            summary.update({prefix + key: sum(r[key] for r in complete) / len(complete) for key in METRICS})
        summaries.append(summary)
    fields = ["dataset", "method", "views", "train_resolution", "eval_resolution", "completed", "expected", "failed",
              *METRICS, *["partial_" + key for key in METRICS]]
    write_csv(BENCH / "dataset_summary.csv", summaries, fields)
    lines = ["# Baseline 实验进度与指标", "", f"状态：{state['status']}。更新：{time.strftime('%Y-%m-%d %H:%M:%S %z')}。", "",
             *(["本轮固定 LLFF/DTU 3-view、Mip-NeRF360 24-view、Blender 8-view。Blender 仅 StereoGS。",
                "训练采用官方实现；Mip-NeRF360 的 TWINGS/StableGS 保持 1/4 训练，统一在 1/8 渲染评测。"]
               if manifest.get("matrix", "first_pass") == "first_pass" else
               ["本轮：LLFF/DTU 的 6-view 与 9-view，三个方法，命令与第一轮相同（仅视角数不同）。",
                "LLFF 6/9-view 的 COLMAP dense 点云由 scripts/colmap_llff_views.py 运行官方 colmap_llff.py 生成。"]),
             "所有主表指标由同一 StereoGS 官方 evaluator 读取 PNG 计算，LPIPS 使用 VGG；DTU 使用前景 mask。",
             "未完成全部场景时仅显示覆盖率，部分场景均值另存 CSV 的 partial_* 列。一次随机种子运行，不是多次运行均值。", "",
             "| 数据集 | 视角 | 方法 | 完成/总数 | 失败 | PSNR | SSIM | LPIPS |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        vals = [f"{row[key]:.4f}" if key in row else "待完成" for key in METRICS]
        lines.append(f"| {row['dataset']} | {row['views']} | {row['method']} | {row['completed']}/{row['expected']} | {row['failed']} | " + " | ".join(vals) + " |")
    lines += ["", "完整参数、官方提交与命令见 manifest.json；每场景结果见 per_scene.csv，执行日志位于 logs/。",
              *(["StableGS scan110 使用官方 --rand_pcd，需在比较中保留这一初始化差异。"]
                if manifest.get("matrix", "first_pass") == "first_pass" else []),
              "StereoGS 使用 dropout=0.3 的完整方法配置；Blender 按官方 README 启用上游批量脚本中注释掉的核心开关。"]
    (BENCH / "RESULTS.md").write_text("\n".join(lines) + "\n")


def run(gpus, retry_failed, selected_index=None, slots=1):
    lock = (BENCH / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read_json(BENCH / "manifest.json")
    state = read_json(BENCH / "status.json")
    state.pop("gpu", None)
    state.pop("current", None)
    state.update(status="running", pid=os.getpid(), gpus=gpus, slots=slots, active={})
    workers = [f"{gpu}.{k}" if slots > 1 else str(gpu) for gpu in gpus for k in range(slots)]
    pending = iter(index for index, status in enumerate(state["jobs"])
                   if (selected_index is None or index == selected_index)
                   and status["status"] not in ("completed", "skipped")
                   and (status["status"] != "failed" or retry_failed))
    active = {}
    exhausted = False

    def publish():
        write_json(BENCH / "status.json", state)
        collect()

    def stop(signum, frame):
        raise KeyboardInterrupt

    def start_step(slot, index):
        job = manifest["jobs"][index]
        status = state["jobs"][index]
        steps = [step for step in job["steps"] if step["name"] not in status["completed_steps"]]
        if not steps:
            status.update(status="completed")
            status.pop("exit_code", None)
            status.pop("failed_step", None)
            state["active"].pop(slot, None)
            return
        step = steps[0]
        log_dir = job_dir(job)
        log_dir.mkdir(parents=True, exist_ok=True)
        stream = (log_dir / f"{step['name']}.log").open("a")
        stream.write("\n" + time.strftime("%Y-%m-%d %H:%M:%S %z") + " " + shlex.join(step["command"]) + "\n")
        stream.flush()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot.split(".")[0], PYTHONUNBUFFERED="1")
        child = subprocess.Popen(step["command"], cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        active[slot] = {"index": index, "child": child, "stream": stream, "started": time.monotonic()}
        status.update(status="running", step=step["name"], gpu=slot, child_pid=child.pid)
        state["active"][slot] = {"index": index, "method": job["method"], "dataset": job["dataset"],
                                     "scene": job["scene"], "step": step["name"]}
        print(f"[GPU {slot}] [{index + 1}/{len(manifest['jobs'])}] {state['active'][slot]}", flush=True)

    signal.signal(signal.SIGTERM, stop)
    try:
        while True:
            changed = False
            for slot, task in list(active.items()):
                code = task["child"].poll()
                if code is None:
                    continue
                task["stream"].close()
                index = task["index"]
                status = state["jobs"][index]
                status["seconds"] += time.monotonic() - task["started"]
                status.pop("child_pid", None)
                del active[slot]
                if code:
                    status.update(status="failed", exit_code=code, failed_step=status["step"])
                    print(f"FAILED {state['active'][slot]}: exit {code}", flush=True)
                    state["active"].pop(slot)
                else:
                    status["completed_steps"].append(status["step"])
                    start_step(slot, index)
                changed = True
            for slot in workers:
                if slot in active or exhausted:
                    continue
                index = next(pending, None)
                if index is None:
                    exhausted = True
                    break
                start_step(slot, index)
                changed = True
            if changed:
                publish()
            if exhausted and not active:
                break
            time.sleep(1)
        if any(s["status"] == "pending" for s in state["jobs"]):
            state["status"] = "paused"
        else:
            state["status"] = "finished_with_failures" if any(s["status"] == "failed" for s in state["jobs"]) else "completed"
    except BaseException:
        state["status"] = "interrupted"
        raise
    finally:
        for slot, task in active.items():
            child = task["child"]
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            child.wait()
            task["stream"].close()
            status = state["jobs"][task["index"]]
            status["seconds"] += time.monotonic() - task["started"]
            status.pop("child_pid", None)
            status["status"] = "pending"
        state["active"] = {}
        state.pop("pid", None)
        publish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run", "score", "collect"])
    parser.add_argument("--gpu", type=int, nargs="+", default=[3])
    parser.add_argument("--index", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--bench", default="first_pass", help="folder under benchmarks/")
    parser.add_argument("--matrix", choices=sorted(MATRICES), default="first_pass", help="plan only")
    parser.add_argument("--slots", type=int, default=1, help="concurrent jobs per GPU")
    args = parser.parse_args()
    BENCH = ROOT / "benchmarks" / args.bench
    if args.action == "plan":
        plan(args.matrix)
    elif args.action == "score":
        score(args.index)
    elif args.action == "collect":
        collect()
    else:
        run(args.gpu, args.retry_failed, args.index, args.slots)
