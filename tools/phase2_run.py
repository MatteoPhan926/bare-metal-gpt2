"""Run one command, retaining stdout/stderr, GPU telemetry and identity receipt.

python tools/phase2_run.py docs/phase2/<unique-name> -- bench/graph_gate.exe gemv
Never overwrites an existing run. GPU workloads must be run serially.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path, skip=0):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(skip)
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def identity():
    tracked = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard"], text=True).splitlines()
    source = {p: sha(p) for p in tracked if Path(p).is_file() and
              (Path(p).suffix in (".cu", ".cuh", ".c", ".cpp", ".h", ".py", ".bat", ".ps1") or p == "PHASE2_PLAN.md")}
    inputs = [Path("weights/gpt2_124m_fp32.bin"), Path("weights/gpt2_124m_int8_kt.bin")]
    inputs += sorted(Path("refdumps").rglob("*.bin")) + [Path("refdumps/meta.json")]
    weights = Path("weights/gpt2_124m_fp32.bin")
    if weights.exists():
        expected = json.loads(Path(str(weights)+".json").read_text())["sha256"]
        if sha(weights, 64) != expected:
            raise RuntimeError("fp32 weight body differs from export manifest")
    return {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
            "source_sha256": source, "input_sha256": {str(p): sha(p) for p in inputs if p.is_file()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("missing command after --")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = identity()
    receipt.update(command=cmd, utc_start=dt.datetime.now(dt.timezone.utc).isoformat(),
                   cwd=str(Path.cwd()), platform=sys.platform,
                   environment={k: v for k, v in os.environ.items() if k.startswith("GPT2_")},
                   executable_sha256=sha(cmd[0]) if Path(cmd[0]).is_file() else None,
                   nvcc_version=subprocess.check_output(["nvcc", "--version"], text=True),
                   external_inputs_sha256={p:sha(p) for p in cmd[1:]
                       if Path(p).suffix==".gguf" and Path(p).is_file()})
    monitor = None
    no_window = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        with (args.output / "telemetry.csv").open("wb") as tele, \
             (args.output / "telemetry.stderr.txt").open("wb") as terr, \
             (args.output / "stdout.txt").open("wb") as out, \
             (args.output / "stderr.txt").open("wb") as err:
            monitor = subprocess.Popen([
                "nvidia-smi", "--query-gpu=timestamp,name,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu",
                "--format=csv", "--loop-ms=200"], stdout=tele, stderr=terr, creationflags=no_window)
            start = time.perf_counter()
            result = subprocess.run(cmd, stdout=out, stderr=err, creationflags=no_window)
            receipt.update(returncode=result.returncode, wall_seconds=time.perf_counter()-start)
    except OSError as error:
        receipt.update(returncode=127, launch_error=str(error))
        raise
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait()
        receipt["utc_end"] = dt.datetime.now(dt.timezone.utc).isoformat()
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    # Full CSV is retained on disk; avoid flooding a terminal with thousands of rows.
    captured = (args.output / "stdout.txt").read_text(errors="replace")
    print("\n".join(line for line in captured.splitlines() if not line.startswith("sample,")))
    print((args.output / "stderr.txt").read_text(errors="replace"), file=sys.stderr)
    print(f"Receipt: {args.output / 'receipt.json'}")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
