import argparse
import csv
import json
import math
from pathlib import Path

from formal_experiment_spec import (
    RESULT_ROOT_NAME,
    build_full_manifests,
    build_phase1_manifests,
    manifest_index,
    select_best_basis_rank,
)


T_CRITICAL_95 = {
    1: None,
    2: 12.706204736432095,
    3: 4.302652729911275,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate formal gated-attention experiment results.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--best-basis-rank", type=int, default=None)
    return parser.parse_args()


def mean_or_none(values):
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def std_or_none(values):
    filtered = [value for value in values if value is not None]
    if len(filtered) < 2:
        return 0.0 if filtered else None
    mean_value = sum(filtered) / len(filtered)
    variance = sum((value - mean_value) ** 2 for value in filtered) / (len(filtered) - 1)
    return math.sqrt(variance)


def average_nullable_lists(list_of_lists):
    if not list_of_lists:
        return []
    list_length = len(list_of_lists[0])
    averaged = []
    for index in range(list_length):
        values = [value_list[index] for value_list in list_of_lists if value_list[index] is not None]
        averaged.append(mean_or_none(values))
    return averaged


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_expected_manifests(manifest_dir: Path, best_basis_rank):
    if manifest_dir is None:
        if best_basis_rank is None:
            manifests = build_phase1_manifests()
        else:
            manifests = build_full_manifests(best_basis_rank)
        return manifest_index(manifests)

    manifests = {}
    for manifest_path in sorted(manifest_dir.glob("*.json")):
        manifest = read_json(manifest_path)
        manifests[manifest["variant_id"]] = manifest
    return manifests


def collect_run_dirs(results_root: Path):
    variant_runs = {}
    for suite_dir in sorted(
        path
        for path in results_root.iterdir()
        if path.is_dir() and path.name not in {"aggregate", "manifests"}
    ):
        for variant_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
            variant_id = variant_dir.name
            variant_runs.setdefault(
                variant_id,
                {
                    "suite": suite_dir.name,
                    "variant_dir": variant_dir,
                    "seed_dirs": {},
                },
            )
            for seed_dir in sorted(path for path in variant_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
                seed = int(seed_dir.name.split("_", 1)[1])
                variant_runs[variant_id]["seed_dirs"][seed] = seed_dir
    return variant_runs


def build_failed_runs(expected_manifests, variant_runs):
    failed_runs = []
    for variant_id, manifest in expected_manifests.items():
        suite_name = manifest["suite_name"]
        expected_seeds = manifest["seeds"]
        variant_entry = variant_runs.get(variant_id)
        for seed in expected_seeds:
            if variant_entry is None or seed not in variant_entry["seed_dirs"]:
                failed_runs.append(
                    {
                        "suite_name": suite_name,
                        "variant_id": variant_id,
                        "seed": seed,
                        "reason": "missing_seed_dir",
                    }
                )
                continue
            seed_dir = variant_entry["seed_dirs"][seed]
            if not (seed_dir / "summary_last.json").exists():
                failed_runs.append(
                    {
                        "suite_name": suite_name,
                        "variant_id": variant_id,
                        "seed": seed,
                        "reason": "missing_summary_last",
                    }
                )
    return failed_runs


def aggregate_variant_record(variant_id, suite_name, seed_summaries):
    val_losses = [summary["val_loss"] for summary in seed_summaries.values()]
    ppls = [summary["ppl"] for summary in seed_summaries.values()]
    train_peak_values = [summary["train_peak_memory_mb"] for summary in seed_summaries.values()]
    eval_peak_values = [summary["eval_peak_memory_mb"] for summary in seed_summaries.values()]
    gate_means = [summary["gate_mean"] for summary in seed_summaries.values()]
    gate_sparsity = [summary["gate_sparsity_02"] for summary in seed_summaries.values()]
    layer_gate_mean = [summary["layer_gate_mean"] for summary in seed_summaries.values()]
    layer_gate_sparsity = [summary["layer_gate_sparsity_02"] for summary in seed_summaries.values()]
    basis_cosine_mean = [summary["basis_offdiag_cosine_mean"] for summary in seed_summaries.values()]
    basis_cosine_max_abs = [summary["basis_offdiag_cosine_max_abs"] for summary in seed_summaries.values()]
    basis_effective_rank = [summary["basis_effective_rank"] for summary in seed_summaries.values()]
    first_summary = next(iter(seed_summaries.values()))

    return {
        "variant_id": variant_id,
        "suite": suite_name,
        "seed_count": len(seed_summaries),
        "available_seeds": sorted(seed_summaries.keys()),
        "val_loss_mean": mean_or_none(val_losses),
        "val_loss_std": std_or_none(val_losses),
        "ppl_mean": mean_or_none(ppls),
        "ppl_std": std_or_none(ppls),
        "extra_param_count": first_summary["extra_param_count"],
        "extra_flops_per_token": first_summary["extra_flops_per_token"],
        "train_peak_memory_mb_mean": mean_or_none(train_peak_values),
        "eval_peak_memory_mb_mean": mean_or_none(eval_peak_values),
        "gate_mean_mean": mean_or_none(gate_means),
        "gate_sparsity_02_mean": mean_or_none(gate_sparsity),
        "layer_gate_mean_mean": average_nullable_lists(layer_gate_mean),
        "layer_gate_sparsity_02_mean": average_nullable_lists(layer_gate_sparsity),
        "basis_offdiag_cosine_mean_mean": average_nullable_lists(basis_cosine_mean),
        "basis_offdiag_cosine_max_abs_mean": average_nullable_lists(basis_cosine_max_abs),
        "basis_effective_rank_mean": average_nullable_lists(basis_effective_rank),
    }


def aggregate_results(expected_manifests, variant_runs):
    aggregated = []
    for variant_id, manifest in expected_manifests.items():
        variant_entry = variant_runs.get(variant_id)
        if variant_entry is None:
            continue
        seed_summaries = {}
        for seed in manifest["seeds"]:
            seed_dir = variant_entry["seed_dirs"].get(seed)
            if seed_dir is None:
                continue
            summary_path = seed_dir / "summary_last.json"
            if not summary_path.exists():
                continue
            seed_summaries[seed] = read_json(summary_path)
        if seed_summaries:
            aggregated.append(aggregate_variant_record(variant_id, manifest["suite_name"], seed_summaries))
    return aggregated


def write_comparison_summary(aggregate_dir: Path, summary_records):
    json_path = aggregate_dir / "comparison_summary.json"
    csv_path = aggregate_dir / "comparison_summary.csv"
    json_path.write_text(json.dumps(summary_records, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for record in summary_records:
        rows.append(
            {
                "variant_id": record["variant_id"],
                "suite": record["suite"],
                "seed_count": record["seed_count"],
                "val_loss_mean": record["val_loss_mean"],
                "val_loss_std": record["val_loss_std"],
                "ppl_mean": record["ppl_mean"],
                "ppl_std": record["ppl_std"],
                "extra_param_count": record["extra_param_count"],
                "extra_flops_per_token": record["extra_flops_per_token"],
                "train_peak_memory_mb_mean": record["train_peak_memory_mb_mean"],
                "eval_peak_memory_mb_mean": record["eval_peak_memory_mb_mean"],
                "gate_mean_mean": record["gate_mean_mean"],
                "gate_sparsity_02_mean": record["gate_sparsity_02_mean"],
                "layer_gate_mean_mean": json.dumps(record["layer_gate_mean_mean"], ensure_ascii=False),
                "layer_gate_sparsity_02_mean": json.dumps(record["layer_gate_sparsity_02_mean"], ensure_ascii=False),
                "basis_offdiag_cosine_mean_mean": json.dumps(record["basis_offdiag_cosine_mean_mean"], ensure_ascii=False),
                "basis_offdiag_cosine_max_abs_mean": json.dumps(record["basis_offdiag_cosine_max_abs_mean"], ensure_ascii=False),
                "basis_effective_rank_mean": json.dumps(record["basis_effective_rank_mean"], ensure_ascii=False),
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["variant_id"])
        writer.writeheader()
        writer.writerows(rows)


def paired_t_interval(deltas):
    if len(deltas) < 2:
        return {
            "delta_mean": mean_or_none(deltas),
            "delta_std": std_or_none(deltas),
            "ci95_low": None,
            "ci95_high": None,
        }
    mean_delta = mean_or_none(deltas)
    std_delta = std_or_none(deltas)
    t_critical = T_CRITICAL_95[len(deltas)]
    half_width = t_critical * std_delta / math.sqrt(len(deltas))
    return {
        "delta_mean": mean_delta,
        "delta_std": std_delta,
        "ci95_low": mean_delta - half_width,
        "ci95_high": mean_delta + half_width,
    }


def build_significance(expected_manifests, variant_runs):
    significance_records = []
    for variant_id, manifest in expected_manifests.items():
        variant_entry = variant_runs.get(variant_id)
        if variant_entry is None:
            continue
        for reference_id in manifest["references"]:
            reference_entry = variant_runs.get(reference_id)
            if reference_entry is None:
                continue
            deltas = []
            seed_win_count = 0
            for seed in manifest["seeds"]:
                variant_summary_path = variant_entry["seed_dirs"].get(seed, Path(".")) / "summary_last.json"
                reference_summary_path = reference_entry["seed_dirs"].get(seed, Path(".")) / "summary_last.json"
                if not variant_summary_path.exists() or not reference_summary_path.exists():
                    continue
                variant_summary = read_json(variant_summary_path)
                reference_summary = read_json(reference_summary_path)
                delta = variant_summary["val_loss"] - reference_summary["val_loss"]
                deltas.append(delta)
                if delta < 0:
                    seed_win_count += 1
            if not deltas:
                continue
            interval = paired_t_interval(deltas)
            significance_records.append(
                {
                    "variant_id": variant_id,
                    "reference_id": reference_id,
                    "delta_mean": interval["delta_mean"],
                    "delta_std": interval["delta_std"],
                    "ci95_low": interval["ci95_low"],
                    "ci95_high": interval["ci95_high"],
                    "seed_win_count": seed_win_count,
                    "is_significantly_better": interval["ci95_high"] is not None and interval["ci95_high"] < 0,
                }
            )
    return significance_records


def write_significance(aggregate_dir: Path, significance_records):
    (aggregate_dir / "significance.json").write_text(
        json.dumps(significance_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_metrics_series(seed_dir: Path):
    metrics_path = seed_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    rows = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_if_possible(plot_name, csv_rows, plot_fn, figures_dir: Path):
    csv_path = figures_dir / f"{plot_name}.csv"
    fieldnames = list(csv_rows[0].keys()) if csv_rows else ["variant_id"]
    write_csv(csv_path, fieldnames, csv_rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    png_path = figures_dir / f"{plot_name}.png"
    plot_fn(plt, png_path)


def generate_baseline_uniform_loss_curves(figures_dir, expected_manifests, variant_runs):
    baseline_variants = [
        manifest["variant_id"]
        for manifest in expected_manifests.values()
        if manifest["suite_name"] == "baseline"
    ]
    series_rows = []
    summary_rows = []
    for variant_id in baseline_variants:
        variant_entry = variant_runs.get(variant_id)
        if variant_entry is None:
            continue
        step_to_values = {}
        for seed_dir in variant_entry["seed_dirs"].values():
            for row in load_metrics_series(seed_dir):
                step_to_values.setdefault(row["step"], []).append(row["val_loss"])
        for step, values in sorted(step_to_values.items()):
            series_rows.append(
                {
                    "variant_id": variant_id,
                    "step": step,
                    "val_loss_mean": mean_or_none(values),
                    "val_loss_std": std_or_none(values),
                }
            )
            summary_rows.append((variant_id, step, mean_or_none(values)))

    def plot_fn(plt, png_path):
        plt.figure(figsize=(10, 6))
        for variant_id in baseline_variants:
            xs = [step for current_variant, step, _ in summary_rows if current_variant == variant_id]
            ys = [value for current_variant, _, value in summary_rows if current_variant == variant_id]
            if xs:
                plt.plot(xs, ys, label=variant_id)
        plt.xlabel("Step")
        plt.ylabel("Validation Loss")
        plt.title("Baseline Uniform Loss Curves")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()

    plot_if_possible("baseline_uniform_loss_curves", series_rows, plot_fn, figures_dir)


def generate_baseline_final_bar(figures_dir, summary_by_variant):
    baseline_ids = [
        "uniform-baseline",
        "uniform-headwise",
        "uniform-groupwise-g2",
        "uniform-groupwise-g4",
        "uniform-groupwise-g8",
        "uniform-groupwise-g16",
        "uniform-groupwise-g32",
        "uniform-elementwise",
    ]
    rows = []
    for variant_id in baseline_ids:
        if variant_id not in summary_by_variant:
            continue
        record = summary_by_variant[variant_id]
        rows.append(
            {
                "variant_id": variant_id,
                "val_loss_mean": record["val_loss_mean"],
                "val_loss_std": record["val_loss_std"],
                "ppl_mean": record["ppl_mean"],
                "ppl_std": record["ppl_std"],
            }
        )

    def plot_fn(plt, png_path):
        plt.figure(figsize=(10, 6))
        xs = range(len(rows))
        plt.bar(xs, [row["val_loss_mean"] for row in rows], yerr=[row["val_loss_std"] for row in rows])
        plt.xticks(list(xs), [row["variant_id"] for row in rows], rotation=45, ha="right")
        plt.ylabel("Final Validation Loss")
        plt.title("Baseline Final Validation Loss")
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()

    plot_if_possible("baseline_final_bar", rows, plot_fn, figures_dir)


def generate_depth_layout_vs_performance(figures_dir, summary_by_variant):
    depth_ids = [variant_id for variant_id in summary_by_variant if variant_id.startswith("depth-")]
    rows = []
    for variant_id in sorted(depth_ids):
        record = summary_by_variant[variant_id]
        rows.append(
            {
                "variant_id": variant_id,
                "val_loss_mean": record["val_loss_mean"],
                "val_loss_std": record["val_loss_std"],
            }
        )

    def plot_fn(plt, png_path):
        plt.figure(figsize=(12, 6))
        xs = range(len(rows))
        plt.bar(xs, [row["val_loss_mean"] for row in rows], yerr=[row["val_loss_std"] for row in rows])
        plt.xticks(list(xs), [row["variant_id"] for row in rows], rotation=45, ha="right")
        plt.ylabel("Final Validation Loss")
        plt.title("Depth-Adaptive Layout Performance")
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()

    plot_if_possible("depth_layout_vs_performance", rows, plot_fn, figures_dir)


def generate_depth_layer_gate_heatmap(figures_dir, summary_by_variant):
    depth_ids = [variant_id for variant_id in summary_by_variant if variant_id.startswith("depth-")]
    rows = []
    for variant_id in sorted(depth_ids):
        record = summary_by_variant[variant_id]
        for layer_index, value in enumerate(record["layer_gate_mean_mean"], start=1):
            rows.append({"variant_id": variant_id, "metric": "layer_gate_mean", "layer": layer_index, "value": value})
        for layer_index, value in enumerate(record["layer_gate_sparsity_02_mean"], start=1):
            rows.append({"variant_id": variant_id, "metric": "layer_gate_sparsity_02", "layer": layer_index, "value": value})

    def plot_fn(plt, png_path):
        import numpy as np

        if not depth_ids:
            return
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        for axis, metric in zip(axes, ["layer_gate_mean", "layer_gate_sparsity_02"]):
            matrix = []
            for variant_id in sorted(depth_ids):
                record = summary_by_variant[variant_id]
                source = record["layer_gate_mean_mean"] if metric == "layer_gate_mean" else record["layer_gate_sparsity_02_mean"]
                matrix.append([float("nan") if value is None else value for value in source])
            image = axis.imshow(np.array(matrix, dtype=float), aspect="auto", interpolation="nearest")
            axis.set_yticks(range(len(depth_ids)))
            axis.set_yticklabels(sorted(depth_ids), fontsize=7)
            axis.set_xticks(range(12))
            axis.set_xticklabels([str(index) for index in range(1, 13)])
            axis.set_title(metric)
            fig.colorbar(image, ax=axis, fraction=0.02, pad=0.01)
        plt.savefig(png_path, dpi=200)
        plt.close()

    plot_if_possible("depth_layer_gate_heatmap", rows, plot_fn, figures_dir)


def generate_basis_rank_vs_performance(figures_dir, summary_by_variant):
    rows = []
    for rank in (4, 8, 16):
        variant_id = f"basis-r{rank}"
        if variant_id not in summary_by_variant:
            continue
        record = summary_by_variant[variant_id]
        rows.append(
            {
                "variant_id": variant_id,
                "val_loss_mean": record["val_loss_mean"],
                "extra_param_count": record["extra_param_count"],
                "extra_flops_per_token": record["extra_flops_per_token"],
            }
        )

    def plot_fn(plt, png_path):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        metrics = ["val_loss_mean", "extra_param_count", "extra_flops_per_token"]
        for axis, metric in zip(axes, metrics):
            axis.bar(range(len(rows)), [row[metric] for row in rows])
            axis.set_xticks(range(len(rows)))
            axis.set_xticklabels([row["variant_id"] for row in rows], rotation=45, ha="right")
            axis.set_title(metric)
        plt.savefig(png_path, dpi=200)
        plt.close()

    plot_if_possible("basis_rank_vs_performance", rows, plot_fn, figures_dir)


def generate_basis_collapse_heatmap(figures_dir, summary_by_variant):
    basis_ids = [variant_id for variant_id in summary_by_variant if variant_id.startswith("basis-r")]
    rows = []
    for variant_id in sorted(basis_ids):
        record = summary_by_variant[variant_id]
        for layer_index, value in enumerate(record["basis_offdiag_cosine_max_abs_mean"], start=1):
            rows.append({"variant_id": variant_id, "metric": "basis_offdiag_cosine_max_abs", "layer": layer_index, "value": value})
        for layer_index, value in enumerate(record["basis_effective_rank_mean"], start=1):
            rows.append({"variant_id": variant_id, "metric": "basis_effective_rank", "layer": layer_index, "value": value})

    def plot_fn(plt, png_path):
        import numpy as np

        if not basis_ids:
            return
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
        for axis, metric in zip(axes, ["basis_offdiag_cosine_max_abs", "basis_effective_rank"]):
            matrix = []
            for variant_id in sorted(basis_ids):
                record = summary_by_variant[variant_id]
                source = (
                    record["basis_offdiag_cosine_max_abs_mean"]
                    if metric == "basis_offdiag_cosine_max_abs"
                    else record["basis_effective_rank_mean"]
                )
                matrix.append([float("nan") if value is None else value for value in source])
            image = axis.imshow(np.array(matrix, dtype=float), aspect="auto", interpolation="nearest")
            axis.set_yticks(range(len(basis_ids)))
            axis.set_yticklabels(sorted(basis_ids))
            axis.set_xticks(range(12))
            axis.set_xticklabels([str(index) for index in range(1, 13)])
            axis.set_title(metric)
            fig.colorbar(image, ax=axis, fraction=0.02, pad=0.01)
        plt.savefig(png_path, dpi=200)
        plt.close()

    plot_if_possible("basis_collapse_heatmap", rows, plot_fn, figures_dir)


def generate_combo_vs_full_basis(figures_dir, summary_by_variant, best_basis_rank):
    variant_ids = [
        f"basis-r{best_basis_rank}",
        "uniform-headwise",
        "depth-E4-H4-N4",
        "depth-G8x4-H4-N4",
        "combo-basis4-H8",
        "combo-basis4-H4-N4",
        "combo-basis6-H6",
        "combo-H8-basis4",
        "combo-basis4-G8x4-N4",
    ]
    rows = []
    for variant_id in variant_ids:
        if variant_id not in summary_by_variant:
            continue
        record = summary_by_variant[variant_id]
        rows.append(
            {
                "variant_id": variant_id,
                "val_loss_mean": record["val_loss_mean"],
                "val_loss_std": record["val_loss_std"],
            }
        )

    def plot_fn(plt, png_path):
        plt.figure(figsize=(12, 6))
        xs = range(len(rows))
        plt.bar(xs, [row["val_loss_mean"] for row in rows], yerr=[row["val_loss_std"] for row in rows])
        plt.xticks(list(xs), [row["variant_id"] for row in rows], rotation=45, ha="right")
        plt.ylabel("Final Validation Loss")
        plt.title("Combination vs Full-Layer Basis")
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()

    plot_if_possible("combo_vs_full_basis", rows, plot_fn, figures_dir)


def generate_figures(aggregate_dir: Path, expected_manifests, variant_runs, summary_records, best_basis_rank):
    figures_dir = aggregate_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_by_variant = {record["variant_id"]: record for record in summary_records}
    generate_baseline_uniform_loss_curves(figures_dir, expected_manifests, variant_runs)
    generate_baseline_final_bar(figures_dir, summary_by_variant)
    generate_depth_layout_vs_performance(figures_dir, summary_by_variant)
    generate_depth_layer_gate_heatmap(figures_dir, summary_by_variant)
    generate_basis_rank_vs_performance(figures_dir, summary_by_variant)
    generate_basis_collapse_heatmap(figures_dir, summary_by_variant)
    if best_basis_rank is not None:
        generate_combo_vs_full_basis(figures_dir, summary_by_variant, best_basis_rank)


def lookup_significance(significance_records, variant_id, reference_id):
    for record in significance_records:
        if record["variant_id"] == variant_id and record["reference_id"] == reference_id:
            return record
    return None


def write_analysis_report(aggregate_dir: Path, summary_records, significance_records, best_basis_rank):
    summary_by_variant = {record["variant_id"]: record for record in summary_records}

    def val_loss_text(variant_id):
        record = summary_by_variant.get(variant_id)
        if record is None:
            return "缺失"
        return f"{record['val_loss_mean']:.6f} ± {record['val_loss_std']:.6f}"

    lines = [
        "# Formal Experiment Analysis Report",
        "",
        f"结果目录：`{aggregate_dir.parent}`",
        f"汇总文件：`{aggregate_dir / 'comparison_summary.json'}`",
        f"显著性文件：`{aggregate_dir / 'significance.json'}`",
        "",
        "## 1. Gate 是否具有 depth-dependent 结构",
        f"- `depth-E4-H4-N4`: {val_loss_text('depth-E4-H4-N4')}",
        f"- `depth-N4-H4-E4`: {val_loss_text('depth-N4-H4-E4')}",
        f"- `depth-random-E4-H4-N4`: {val_loss_text('depth-random-E4-H4-N4')}",
        f"- `depth-G8x4-H4-N4`: {val_loss_text('depth-G8x4-H4-N4')}",
        f"- `depth-N4-H4-G8x4`: {val_loss_text('depth-N4-H4-G8x4')}",
        f"- `depth-random-G8x4-H4-N4`: {val_loss_text('depth-random-G8x4-H4-N4')}",
        "- 相关逐层统计见 `figures/depth_layer_gate_heatmap.png` 与 `figures/depth_layer_gate_heatmap.csv`。",
        "",
        "## 2. 细粒度 gate 是否主要在浅层有效",
        f"- `depth-E4-H4-N4` vs `depth-N4-H4-E4`: {json.dumps(lookup_significance(significance_records, 'depth-E4-H4-N4', 'depth-N4-H4-E4'), ensure_ascii=False)}",
        f"- `combo-basis4-H8` vs `combo-H8-basis4`: {val_loss_text('combo-basis4-H8')} vs {val_loss_text('combo-H8-basis4')}",
        f"- `combo-basis4-H4-N4` vs `combo-H8-basis4`: {val_loss_text('combo-basis4-H4-N4')} vs {val_loss_text('combo-H8-basis4')}",
        "",
        "## 3. Groupwise 表现不佳的原因",
        f"- `uniform-groupwise-g2`: {val_loss_text('uniform-groupwise-g2')}",
        f"- `uniform-groupwise-g4`: {val_loss_text('uniform-groupwise-g4')}",
        f"- `uniform-groupwise-g8`: {val_loss_text('uniform-groupwise-g8')}",
        f"- `uniform-groupwise-g16`: {val_loss_text('uniform-groupwise-g16')}",
        f"- `uniform-groupwise-g32`: {val_loss_text('uniform-groupwise-g32')}",
        (
            f"- `basis-r{best_basis_rank}` vs `uniform-groupwise-g8`: "
            f"{json.dumps(lookup_significance(significance_records, f'basis-r{best_basis_rank}', 'uniform-groupwise-g8'), ensure_ascii=False)}"
            if best_basis_rank is not None
            else "- 尚未选出 `r*`。"
        ),
        "",
        "## 4. Low-rank 是否优于手工分组",
        f"- `basis-r4`: {val_loss_text('basis-r4')}",
        f"- `basis-r8`: {val_loss_text('basis-r8')}",
        f"- `basis-r16`: {val_loss_text('basis-r16')}",
        f"- `uniform-headwise`: {val_loss_text('uniform-headwise')}",
        f"- `uniform-groupwise-g8`: {val_loss_text('uniform-groupwise-g8')}",
        f"- `uniform-elementwise`: {val_loss_text('uniform-elementwise')}",
        "- Basis collapse 相关统计见 `figures/basis_collapse_heatmap.png` 与 `comparison_summary.json` 中的 `basis_*` 字段。",
        "",
        "## 5. 表达能力与稳定性的权衡",
    ]

    if best_basis_rank is not None:
        lines.extend(
            [
                f"- `basis-r{best_basis_rank}`: {val_loss_text(f'basis-r{best_basis_rank}')}",
                f"- `basis-r{best_basis_rank}-l2norm`: {val_loss_text(f'basis-r{best_basis_rank}-l2norm')}",
                f"- `basis-r{best_basis_rank}-tau0p7`: {val_loss_text(f'basis-r{best_basis_rank}-tau0p7')}",
                f"- `basis-r{best_basis_rank}-tau1p3`: {val_loss_text(f'basis-r{best_basis_rank}-tau1p3')}",
            ]
        )
    lines.extend(
        [
            "- 训练/评估显存、额外 FLOPs 与失败记录见 `comparison_summary.json`、`significance.json` 与 `failed_runs.json`。",
            "",
            "## 6. 参考图表",
            "- `figures/baseline_uniform_loss_curves.png`",
            "- `figures/baseline_final_bar.png`",
            "- `figures/depth_layout_vs_performance.png`",
            "- `figures/depth_layer_gate_heatmap.png`",
            "- `figures/basis_rank_vs_performance.png`",
            "- `figures/basis_collapse_heatmap.png`",
            "- `figures/combo_vs_full_basis.png`",
        ]
    )

    (aggregate_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main():
    args = parse_args()
    results_root = args.results_root
    aggregate_dir = results_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    expected_manifests = load_expected_manifests(args.manifest_dir, args.best_basis_rank)
    variant_runs = collect_run_dirs(results_root)
    failed_runs = build_failed_runs(expected_manifests, variant_runs)
    (aggregate_dir / "failed_runs.json").write_text(
        json.dumps(failed_runs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_records = aggregate_results(expected_manifests, variant_runs)
    write_comparison_summary(aggregate_dir, summary_records)

    best_basis_rank = args.best_basis_rank
    if best_basis_rank is None:
        basis_candidates = [record for record in summary_records if record["variant_id"] in {"basis-r4", "basis-r8", "basis-r16"}]
        if len(basis_candidates) == 3:
            best_basis_rank = select_best_basis_rank(basis_candidates)

    significance_records = build_significance(expected_manifests, variant_runs)
    write_significance(aggregate_dir, significance_records)
    generate_figures(aggregate_dir, expected_manifests, variant_runs, summary_records, best_basis_rank)
    write_analysis_report(aggregate_dir, summary_records, significance_records, best_basis_rank)


if __name__ == "__main__":
    main()
