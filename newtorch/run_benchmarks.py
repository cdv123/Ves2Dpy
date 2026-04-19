import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


CSV_COLUMNS = [
    "use_torchrun",
    "input_file",
    "output_file",
    "scp_file",
    "farField",
    "dt",
    "T",
    "wT",
    "nv",
    "N",
    "pr",
    "vortexSize",
    "chanWidth",
    "farFieldSpeed",
    "nProcs",
    "time",
]


def parse_bool(value):
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def nonempty(value):
    return value is not None and str(value).strip() != ""


def load_rows(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    missing = [c for c in CSV_COLUMNS if c not in reader.fieldnames]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    return rows, reader.fieldnames


def save_rows(csv_path, rows, fieldnames):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_command(row, target_script, python_executable):
    cmd = []

    use_torchrun = parse_bool(row.get("use_torchrun"))
    nprocs = str(row.get("nProcs", "")).strip() or "1"

    if use_torchrun:
        cmd.extend(["torchrun", "--nproc_per_node", nprocs, target_script])
    else:
        cmd.extend([python_executable, target_script])

    # Map CSV columns to your argparse flags
    if nonempty(row.get("dt")):
        cmd.extend(["--dt", str(row["dt"]).strip()])
    if nonempty(row.get("T")):
        cmd.extend(["--T", str(row["T"]).strip()])
    if nonempty(row.get("wT")):
        cmd.extend(["--wT", str(row["wT"]).strip()])
    if nonempty(row.get("N")):
        cmd.extend(["--N", str(row["N"]).strip()])
    if nonempty(row.get("nv")):
        cmd.extend(["--nv", str(row["nv"]).strip()])
    if nonempty(row.get("pr")):
        cmd.extend(["--pr", str(row["pr"]).strip()])

    if nonempty(row.get("farField")):
        cmd.extend(["--farField", str(row["farField"]).strip()])
    if nonempty(row.get("vortexSize")):
        cmd.extend(["--vs", str(row["vortexSize"]).strip()])
    if nonempty(row.get("chanWidth")):
        cmd.extend(["--cw", str(row["chanWidth"]).strip()])
    if nonempty(row.get("farFieldSpeed")):
        cmd.extend(["--speed", str(row["farFieldSpeed"]).strip()])

    if nonempty(row.get("input_file")):
        cmd.extend(["--input", str(row["input_file"]).strip()])
    if nonempty(row.get("output_file")):
        cmd.extend(["--output", str(row["output_file"]).strip()])

    return cmd


def try_scp(remote_path, local_dest):
    """
    Best-effort SCP:
    - If scp is not installed, skip.
    - If remote_path is empty, skip.
    - If local_dest is empty, copy into current directory using the remote basename.
    """
    if not nonempty(remote_path):
        return False, "scp_file empty"

    if shutil.which("scp") is None:
        return False, "scp not available"

    remote_path = str(remote_path).strip()

    if nonempty(local_dest):
        local_dest = str(local_dest).strip()
        local_parent = Path(local_dest).expanduser().resolve().parent
        local_parent.mkdir(parents=True, exist_ok=True)
    else:
        local_dest = Path(remote_path.split(":")[-1]).name

    result = subprocess.run(
        ["scp", remote_path, local_dest],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode == 0:
        return True, f"copied to {local_dest}"
    return False, result.stderr.strip() or "scp failed"


def run_one(row, target_script, python_executable, working_dir=None, dry_run=False):
    cmd = build_command(row, target_script, python_executable)
    print("Running:", " ".join(cmd), flush=True)

    if dry_run:
        return {
            "returncode": 0,
            "elapsed": 0.0,
            "stdout": "",
            "stderr": "",
            "scp_ok": False,
            "scp_msg": "dry-run",
        }

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=working_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    scp_ok, scp_msg = try_scp(row.get("scp_file"), row.get("output_file"))

    return {
        "returncode": proc.returncode,
        "elapsed": elapsed,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "scp_ok": scp_ok,
        "scp_msg": scp_msg,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run vesicle simulations from a CSV, time them, and update the CSV."
    )
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument(
        "--target-script",
        required=True,
        help="Python simulation script to run for each CSV row",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use when use_torchrun is false",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Optional working directory for the simulation command",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip rows where the 'time' column is already non-empty",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if a run fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    args = parser.parse_args()

    rows, fieldnames = load_rows(args.csv_file)

    for i, row in enumerate(rows):
        if args.skip_completed and nonempty(row.get("time")):
            print(f"[row {i}] skipping because time is already set: {row['time']}")
            continue

        result = run_one(
            row=row,
            target_script=args.target_script,
            python_executable=args.python,
            working_dir=args.workdir,
            dry_run=args.dry_run,
        )

        row["time"] = f"{result['elapsed']:.6f}"
        save_rows(args.csv_file, rows, fieldnames)

        print(f"[row {i}] return code: {result['returncode']}")
        print(f"[row {i}] elapsed: {result['elapsed']:.6f} s")
        print(f"[row {i}] scp: {result['scp_msg']}")

        if result["stdout"].strip():
            print(f"[row {i}] stdout:\n{result['stdout']}")
        if result["stderr"].strip():
            print(f"[row {i}] stderr:\n{result['stderr']}", file=sys.stderr)

        if result["returncode"] != 0 and args.stop_on_error:
            print(f"[row {i}] stopping due to error", file=sys.stderr)
            sys.exit(result["returncode"])

    print("Done.")


if __name__ == "__main__":
    main()
