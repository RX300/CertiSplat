"""Skip one method in a running benchmark queue without restarting it (a restart kills the active jobs).

    python scripts/skip_method.py watch --bench views_69 --method StableGS   # while the runner is alive
    python scripts/skip_method.py mark  --bench views_69 --method StableGS   # after it has exited

watch: every time the runner starts a step of a job of that method, its process group is terminated, so the
runner records the job as failed and moves on (seconds per job). mark: rewrites those failed jobs as
"skipped" (completed jobs are kept) and refreshes the summaries.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=["watch", "mark"])
parser.add_argument("--bench", required=True)
parser.add_argument("--method", required=True)
args = parser.parse_args()
bench = ROOT / "benchmarks" / args.bench
manifest = json.loads((bench / "manifest.json").read_text())["jobs"]


def runner_alive(state):
    try:
        os.kill(state.get("pid", 0), 0)
        return state.get("pid", 0) > 0
    except OSError:
        return False


if args.action == "watch":
    killed = set()
    while True:
        try:
            state = json.loads((bench / "status.json").read_text())
        except (json.JSONDecodeError, OSError):
            time.sleep(1)
            continue
        if not runner_alive(state):
            print("runner exited; watch ends", flush=True)
            break
        for job, status in zip(manifest, state["jobs"]):
            pid = status.get("child_pid")
            if job["method"] == args.method and status["status"] == "running" and pid and pid not in killed:
                try:
                    os.killpg(pid, signal.SIGTERM)
                    print(time.strftime("%H:%M:%S"), "skipped", job["dataset"], job["views"], job["scene"], flush=True)
                except ProcessLookupError:
                    pass
                killed.add(pid)
        time.sleep(2)
else:
    state = json.loads((bench / "status.json").read_text())
    if runner_alive(state):
        sys.exit("runner still running; mark after it exits")
    n = 0
    for job, status in zip(manifest, state["jobs"]):
        if job["method"] == args.method and status["status"] != "completed":
            status.update(status="skipped")
            for key in ("exit_code", "failed_step", "step", "gpu", "child_pid"):
                status.pop(key, None)
            n += 1
    (bench / "status.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/benchmark.py"), "collect", "--bench", args.bench], check=True)
    print(f"marked {n} {args.method} jobs as skipped")
