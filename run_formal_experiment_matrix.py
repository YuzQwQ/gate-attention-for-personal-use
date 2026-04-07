import argparse
import json
import subprocess
import sys
from pathlib import Path

from formal_experiment_spec import (
    RESULT_ROOT_NAME,
    build_phase1_manifests,
    build_phase2_manifests,
    select_best_basis_rank,
    write_manifests,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full formal gated-attention experiment matrix.")
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--results-base-dir", type=Path, default=Path("results"))
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--prepare-assets", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_command(command, cwd, stdout_path=None, stderr_path=None):
    stdout_handle = stdout_path.open("w", encoding="utf-8") if stdout_path is not None else None
    stderr_handle = stderr_path.open("w", encoding="utf-8") if stderr_path is not None else None
    try:
        result = subprocess.run(command, cwd=cwd, stdout=stdout_handle, stderr=stderr_handle)
    finally:
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()
    return result.returncode


def maybe_prepare_assets(args, repo_root: Path):
    train_file = args.assets_dir / "train.txt"
    valid_file = args.assets_dir / "valid.txt"
    tokenizer_dir = args.assets_dir / "tokenizer"
    if train_file.exists() and valid_file.exists() and tokenizer_dir.exists() and not args.prepare_assets:
        return

    command = [
        args.python_exe,
        str(repo_root / "prepare_wikitext103_assets.py"),
        "--output-dir",
        str(args.assets_dir),
    ]
    subprocess.run(command, cwd=repo_root, check=True)


def run_manifest_suite(args, repo_root: Path, results_root: Path, manifest_paths):
    train_file = args.assets_dir / "train.txt"
    valid_file = args.assets_dir / "valid.txt"
    tokenizer_dir = args.assets_dir / "tokenizer"

    for variant_id, manifest_path in manifest_paths.items():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        suite_name = manifest["suite_name"]
        for seed in manifest["seeds"]:
            seed_dir = results_root / suite_name / variant_id / f"seed_{seed}"
            summary_last_path = seed_dir / "summary_last.json"
            if args.skip_existing and summary_last_path.exists():
                continue

            seed_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = seed_dir / "stdout.log"
            stderr_path = seed_dir / "stderr.log"
            command = [
                args.python_exe,
                str(repo_root / "train_formal_experiment.py"),
                "--manifest",
                str(manifest_path),
                "--train-file",
                str(train_file),
                "--valid-file",
                str(valid_file),
                "--tokenizer-name-or-path",
                str(tokenizer_dir),
                "--results-root",
                str(results_root),
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--retry-count",
                "0",
            ]
            return_code = run_command(command, repo_root, stdout_path, stderr_path)
            if return_code != 0:
                retry_command = command[:-1] + ["1"]
                return_code = run_command(retry_command, repo_root, stdout_path, stderr_path)
                if return_code != 0:
                    print(
                        json.dumps(
                            {
                                "variant_id": variant_id,
                                "suite_name": suite_name,
                                "seed": seed,
                                "status": "failed",
                            },
                            ensure_ascii=False,
                        )
                    )


def run_aggregate(args, repo_root: Path, results_root: Path, manifest_dir: Path, best_basis_rank=None):
    command = [
        args.python_exe,
        str(repo_root / "aggregate_formal_experiments.py"),
        "--results-root",
        str(results_root),
        "--manifest-dir",
        str(manifest_dir),
    ]
    if best_basis_rank is not None:
        command.extend(["--best-basis-rank", str(best_basis_rank)])
    subprocess.run(command, cwd=repo_root, check=True)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    results_root = args.results_base_dir / RESULT_ROOT_NAME
    manifest_dir = args.manifest_dir or (results_root / "manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    maybe_prepare_assets(args, repo_root)

    phase1_manifest_dir = manifest_dir / "phase1"
    phase1_manifests = build_phase1_manifests()
    phase1_paths = write_manifests(phase1_manifests, phase1_manifest_dir)
    run_manifest_suite(args, repo_root, results_root, phase1_paths)
    run_aggregate(args, repo_root, results_root, phase1_manifest_dir)

    comparison_summary_path = results_root / "aggregate" / "comparison_summary.json"
    comparison_summary = json.loads(comparison_summary_path.read_text(encoding="utf-8"))
    best_basis_rank = select_best_basis_rank(
        [record for record in comparison_summary if record["variant_id"] in {"basis-r4", "basis-r8", "basis-r16"}]
    )
    (results_root / "aggregate" / "selected_basis_rank.json").write_text(
        json.dumps({"best_basis_rank": best_basis_rank}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    phase2_manifest_dir = manifest_dir / "phase2"
    phase2_manifests = build_phase2_manifests(best_basis_rank)
    phase2_paths = write_manifests(phase2_manifests, phase2_manifest_dir)
    run_manifest_suite(args, repo_root, results_root, phase2_paths)

    full_manifest_dir = manifest_dir / "all"
    write_manifests(phase1_manifests + phase2_manifests, full_manifest_dir)
    run_aggregate(args, repo_root, results_root, full_manifest_dir, best_basis_rank=best_basis_rank)


if __name__ == "__main__":
    main()
