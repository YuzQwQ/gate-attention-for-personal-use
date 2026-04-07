import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from formal_experiment_spec import FIXED_SEEDS
from train_first_round_compare import (
    TokenBlockDataset,
    build_token_blocks,
    load_local_qwen_modules,
    load_tokenizer_compat,
    make_scheduler,
    resolve_special_token_id,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run one formal gated-attention experiment from a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--tokenizer-name-or-path", type=str, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=FIXED_SEEDS)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--retry-count", type=int, default=0)
    return parser.parse_args()


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


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


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_manifest(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def hash_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_path(path: Path):
    if path.is_file():
        return hash_file(path)

    digest = hashlib.sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(hash_file(file_path).encode("utf-8"))
    return digest.hexdigest()


def build_datasets(train_file: Path, valid_file: Path, tokenizer, context_length: int):
    train_blocks = build_token_blocks(tokenizer, train_file, context_length)
    valid_blocks = build_token_blocks(tokenizer, valid_file, context_length)
    return TokenBlockDataset(train_blocks), TokenBlockDataset(valid_blocks)


def build_config(manifest, tokenizer, config_cls):
    model_config = manifest["model_config"]
    pad_token_id = resolve_special_token_id(tokenizer.pad_token_id, tokenizer.eos_token_id)
    if pad_token_id is None:
        pad_token_id = 0
    eos_token_id = resolve_special_token_id(tokenizer.eos_token_id, pad_token_id)
    bos_token_id = resolve_special_token_id(tokenizer.bos_token_id, eos_token_id)

    rope_scaling = None
    if "default" not in ROPE_INIT_FUNCTIONS:
        rope_scaling = {"rope_type": "linear", "factor": 1.0}

    return config_cls(
        vocab_size=len(tokenizer),
        hidden_size=model_config["hidden_size"],
        intermediate_size=model_config["intermediate_size"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_attention_heads=model_config["num_attention_heads"],
        num_key_value_heads=model_config["num_key_value_heads"],
        head_dim=model_config["head_dim"],
        max_position_embeddings=model_config["context_length"],
        attention_bias=model_config["qkv_bias"],
        qkv_bias=model_config["qkv_bias"],
        attention_dropout=model_config["attention_dropout"],
        use_qk_norm=model_config["use_qk_norm"],
        rope_scaling=rope_scaling,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        use_cache=False,
        _attn_implementation="eager",
        layer_gate_layout=manifest["layer_gate_layout"],
    )


def build_baseline_config(manifest, tokenizer, config_cls):
    baseline_manifest = dict(manifest)
    baseline_manifest["layer_gate_layout"] = [
        {
            "gate_type": "none",
            "num_gate_groups": None,
            "basis_rank": None,
            "basis_alpha_norm": False,
            "basis_temperature": 1.0,
        }
        for _ in range(manifest["model_config"]["num_hidden_layers"])
    ]
    return build_config(baseline_manifest, tokenizer, config_cls)


def compute_gate_extra_flops_per_token(manifest):
    hidden_size = manifest["model_config"]["hidden_size"]
    num_heads = manifest["model_config"]["num_attention_heads"]
    head_dim = manifest["model_config"]["head_dim"]
    extra_flops = 0

    for layer_spec in manifest["layer_gate_layout"]:
        gate_type = layer_spec["gate_type"]
        if gate_type == "none":
            continue
        if gate_type == "shared_headwise":
            num_gate_groups = 1
            extra_gate_dims = num_heads * num_gate_groups
            extra_flops += 2 * hidden_size * extra_gate_dims
            extra_flops += 4 * extra_gate_dims
            extra_flops += num_heads * head_dim
        elif gate_type == "shared_groupwise":
            num_gate_groups = layer_spec["num_gate_groups"]
            extra_gate_dims = num_heads * num_gate_groups
            extra_flops += 2 * hidden_size * extra_gate_dims
            extra_flops += 4 * extra_gate_dims
            extra_flops += num_heads * head_dim
        elif gate_type == "shared_elementwise":
            extra_gate_dims = num_heads * head_dim
            extra_flops += 2 * hidden_size * extra_gate_dims
            extra_flops += 4 * extra_gate_dims
            extra_flops += num_heads * head_dim
        elif gate_type == "basis":
            rank = layer_spec["basis_rank"]
            extra_flops += num_heads * (2 * head_dim * rank)
            if layer_spec["basis_alpha_norm"]:
                extra_flops += num_heads * (3 * rank)
            extra_flops += num_heads * (2 * rank * head_dim)
            extra_flops += num_heads * (4 * head_dim)
            extra_flops += num_heads * head_dim
        else:
            raise ValueError(f"Unsupported gate type for FLOPs accounting: {gate_type}")

    return extra_flops


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def reset_cuda_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def current_peak_memory_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def collect_gate_metrics(model):
    layer_gate_mean = []
    layer_gate_std = []
    layer_gate_sparsity_02 = []
    basis_offdiag_cosine_mean = []
    basis_offdiag_cosine_max_abs = []
    basis_effective_rank = []

    for layer in model.model.layers:
        stats = getattr(layer.self_attn, "last_gate_stats", None)
        if not stats:
            layer_gate_mean.append(None)
            layer_gate_std.append(None)
            layer_gate_sparsity_02.append(None)
            basis_offdiag_cosine_mean.append(None)
            basis_offdiag_cosine_max_abs.append(None)
            basis_effective_rank.append(None)
            continue

        layer_gate_mean.append(stats.get("mean"))
        layer_gate_std.append(stats.get("std"))
        layer_gate_sparsity_02.append(stats.get("sparsity_02"))
        basis_offdiag_cosine_mean.append(stats.get("basis_offdiag_cosine_mean"))
        basis_offdiag_cosine_max_abs.append(stats.get("basis_offdiag_cosine_max_abs"))
        basis_effective_rank.append(stats.get("basis_effective_rank"))

    return {
        "gate_mean": mean_or_none(layer_gate_mean),
        "gate_std": mean_or_none(layer_gate_std),
        "gate_sparsity_02": mean_or_none(layer_gate_sparsity_02),
        "layer_gate_mean": layer_gate_mean,
        "layer_gate_std": layer_gate_std,
        "layer_gate_sparsity_02": layer_gate_sparsity_02,
        "basis_offdiag_cosine_mean": basis_offdiag_cosine_mean,
        "basis_offdiag_cosine_max_abs": basis_offdiag_cosine_max_abs,
        "basis_effective_rank": basis_effective_rank,
    }


def evaluate(model, valid_loader, device, max_eval_batches):
    model.eval()
    losses = []
    gate_means = []
    gate_stds = []
    gate_sparsity_values = []
    layer_gate_mean_batches = []
    layer_gate_std_batches = []
    layer_gate_sparsity_batches = []
    basis_cosine_mean_batches = []
    basis_cosine_max_abs_batches = []
    basis_effective_rank_batches = []

    reset_cuda_peak_memory()
    with torch.no_grad():
        for batch_index, batch in enumerate(valid_loader):
            if max_eval_batches > 0 and batch_index >= max_eval_batches:
                break

            batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
                output_attentions=False,
            )
            losses.append(outputs.loss.item())

            gate_metrics = collect_gate_metrics(model)
            gate_means.append(gate_metrics["gate_mean"])
            gate_stds.append(gate_metrics["gate_std"])
            gate_sparsity_values.append(gate_metrics["gate_sparsity_02"])
            layer_gate_mean_batches.append(gate_metrics["layer_gate_mean"])
            layer_gate_std_batches.append(gate_metrics["layer_gate_std"])
            layer_gate_sparsity_batches.append(gate_metrics["layer_gate_sparsity_02"])
            basis_cosine_mean_batches.append(gate_metrics["basis_offdiag_cosine_mean"])
            basis_cosine_max_abs_batches.append(gate_metrics["basis_offdiag_cosine_max_abs"])
            basis_effective_rank_batches.append(gate_metrics["basis_effective_rank"])

    val_loss = mean_or_none(losses)
    ppl = None if val_loss is None else math.exp(min(val_loss, 20.0))
    return {
        "val_loss": val_loss,
        "ppl": ppl,
        "gate_mean": mean_or_none(gate_means),
        "gate_std": mean_or_none(gate_stds),
        "gate_sparsity_02": mean_or_none(gate_sparsity_values),
        "layer_gate_mean": average_nullable_lists(layer_gate_mean_batches),
        "layer_gate_std": average_nullable_lists(layer_gate_std_batches),
        "layer_gate_sparsity_02": average_nullable_lists(layer_gate_sparsity_batches),
        "basis_offdiag_cosine_mean": average_nullable_lists(basis_cosine_mean_batches),
        "basis_offdiag_cosine_max_abs": average_nullable_lists(basis_cosine_max_abs_batches),
        "basis_effective_rank": average_nullable_lists(basis_effective_rank_batches),
        "eval_peak_memory_mb": current_peak_memory_mb(),
    }


def git_commit(repo_root: Path):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def train_one_run(args, manifest, repo_root):
    set_seed(args.seed)
    config_cls, model_cls = load_local_qwen_modules(repo_root)
    tokenizer = load_tokenizer_compat(args.tokenizer_name_or_path)
    train_dataset, valid_dataset = build_datasets(
        args.train_file,
        args.valid_file,
        tokenizer,
        manifest["model_config"]["context_length"],
    )

    config = build_config(manifest, tokenizer, config_cls)
    model = model_cls(config).to(args.device)
    optimizer = AdamW(
        model.parameters(),
        lr=manifest["training_config"]["learning_rate"],
        weight_decay=manifest["training_config"]["weight_decay"],
    )
    scheduler = make_scheduler(
        optimizer,
        manifest["training_config"]["warmup_steps"],
        manifest["training_config"]["max_steps"],
    )

    baseline_model = model_cls(build_baseline_config(manifest, tokenizer, config_cls))
    total_param_count = count_parameters(model)
    baseline_param_count = count_parameters(baseline_model)
    del baseline_model

    extra_param_count = total_param_count - baseline_param_count
    extra_flops_per_token = compute_gate_extra_flops_per_token(manifest)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=manifest["training_config"]["batch_size"],
        shuffle=True,
        generator=train_generator,
        num_workers=manifest["training_config"]["num_workers"],
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=manifest["training_config"]["eval_batch_size"],
        shuffle=False,
        num_workers=manifest["training_config"]["num_workers"],
        drop_last=False,
    )

    seed_dir = (
        args.results_root
        / manifest["suite_name"]
        / manifest["variant_id"]
        / f"seed_{args.seed}"
    )
    final_model_dir = seed_dir / "final_model"
    seed_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)
    for log_name in ("stdout.log", "stderr.log"):
        log_path = seed_dir / log_name
        if not log_path.exists():
            log_path.touch()

    config.save_pretrained(seed_dir)
    (seed_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics_path = seed_dir / "metrics.jsonl"
    best_record = None
    last_record = None
    all_train_peak_values = []
    all_eval_peak_values = []
    running_loss = 0.0
    running_steps = 0
    interval_step_times = []
    train_iter = iter(train_loader)

    metadata = {
        "dataset_name": manifest["dataset_name"],
        "dataset_config_name": manifest["dataset_config_name"],
        "dataset_splits": ["train", "validation"],
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "tokenizer_hash": hash_path(Path(args.tokenizer_name_or_path)),
        "train_file": str(args.train_file),
        "valid_file": str(args.valid_file),
        "seed": args.seed,
        "device": args.device,
        "git_commit": git_commit(repo_root),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers_version": __import__("transformers").__version__,
        "start_time_utc": utc_now_iso(),
        "retry_count": args.retry_count,
        "manifest_path": str(args.manifest),
    }

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        reset_cuda_peak_memory()
        for step in range(1, manifest["training_config"]["max_steps"] + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            model.train()
            batch = move_batch_to_device(batch, args.device)
            step_start = time.perf_counter()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            loss = outputs.loss
            loss.backward()

            max_grad_norm = manifest["training_config"]["max_grad_norm"]
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            interval_step_times.append(time.perf_counter() - step_start)
            running_loss += loss.item()
            running_steps += 1

            if step % manifest["training_config"]["eval_interval"] == 0 or step == manifest["training_config"]["max_steps"]:
                train_peak_memory_mb = current_peak_memory_mb()
                eval_metrics = evaluate(
                    model,
                    valid_loader,
                    args.device,
                    manifest["training_config"]["max_eval_batches"],
                )
                all_train_peak_values.append(train_peak_memory_mb)
                all_eval_peak_values.append(eval_metrics["eval_peak_memory_mb"])

                last_record = {
                    "step": step,
                    "seed": args.seed,
                    "suite_name": manifest["suite_name"],
                    "variant_id": manifest["variant_id"],
                    "train_loss_window": running_loss / max(1, running_steps),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "avg_step_time": mean_or_none(interval_step_times),
                    "total_param_count": total_param_count,
                    "extra_param_count": extra_param_count,
                    "extra_flops_per_token": extra_flops_per_token,
                    "train_peak_memory_mb": train_peak_memory_mb,
                    "git_commit": metadata["git_commit"],
                    "device": args.device,
                    **eval_metrics,
                }
                metrics_file.write(json.dumps(last_record, ensure_ascii=False) + "\n")
                metrics_file.flush()

                if best_record is None or last_record["val_loss"] < best_record["val_loss"]:
                    best_record = dict(last_record)

                running_loss = 0.0
                running_steps = 0
                interval_step_times = []
                reset_cuda_peak_memory()

    metadata["end_time_utc"] = utc_now_iso()
    (seed_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (seed_dir / "summary_last.json").write_text(
        json.dumps(last_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (seed_dir / "summary_best.json").write_text(
        json.dumps(best_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    resource_metrics = {
        "variant_id": manifest["variant_id"],
        "suite_name": manifest["suite_name"],
        "total_param_count": total_param_count,
        "baseline_param_count": baseline_param_count,
        "extra_param_count": extra_param_count,
        "extra_flops_per_token": extra_flops_per_token,
        "train_peak_memory_mb_last": last_record["train_peak_memory_mb"],
        "eval_peak_memory_mb_last": last_record["eval_peak_memory_mb"],
        "train_peak_memory_mb_global": max(all_train_peak_values) if all_train_peak_values else 0.0,
        "eval_peak_memory_mb_global": max(all_eval_peak_values) if all_eval_peak_values else 0.0,
    }
    (seed_dir / "resource_metrics.json").write_text(
        json.dumps(resource_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    model.save_pretrained(final_model_dir)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    manifest = load_manifest(args.manifest)
    train_one_run(args, manifest, repo_root)


if __name__ == "__main__":
    main()
