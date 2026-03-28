import argparse
import importlib.util
import json
import math
import random
import re
import site
import sys
import time
import types
from pathlib import Path

USER_SITE = site.getusersitepackages()
if USER_SITE not in sys.path:
    sys.path.append(USER_SITE)

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer  # noqa: E402
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS  # noqa: E402


DEFAULT_VARIANTS = ("baseline", "shared-headwise", "a-headwise")
SHARED_GROUPWISE_VARIANTS = (
    "baseline",
    "shared-headwise",
    "shared-groupwise-g4",
    "shared-groupwise-g8",
    "shared-elementwise",
)
RESIDUAL_G8_VARIANTS = (
    "shared-groupwise-g8",
    "shared-groupwise-g8-alpha0p02",
    "shared-groupwise-g8-alpha0p05",
    "shared-groupwise-g8-alpha0p08",
)
TEMPERATURE_G8_VARIANTS = (
    "shared-groupwise-g8-tau0p7",
    "shared-groupwise-g8-tau1p0",
    "shared-groupwise-g8-tau1p3",
)
HYBRID_G8_VARIANTS = (
    "shared-headwise",
    "shared-groupwise-g8",
    "shared-hybrid-g8-lambda0p25",
    "shared-hybrid-g8-lambda0p5",
    "shared-hybrid-g8-lambda0p75",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run attention-gating comparison experiments with local Qwen3 modules."
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--tokenizer-name-or-path", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--num-key-value-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--max-position-embeddings", type=int, default=0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--qkv-bias", action="store_true")
    parser.add_argument("--use-qk-norm", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help=(
            "Variant names to run. Supported values include baseline, shared-headwise, "
            f"shared-elementwise, a-headwise, and shared-groupwise-g<N>. "
            f"Recommended shared-groupwise sweep: {', '.join(SHARED_GROUPWISE_VARIANTS)}. "
            f"Recommended residual g8 sweep: {', '.join(RESIDUAL_G8_VARIANTS)}. "
            f"Recommended temperature g8 sweep: {', '.join(TEMPERATURE_G8_VARIANTS)}. "
            f"Recommended hybrid g8 sweep: {', '.join(HYBRID_G8_VARIANTS)}."
        ),
    )
    return parser.parse_args()


def ensure_transformers_compat():
    import transformers.cache_utils as cache_utils

    if not hasattr(cache_utils, "SlidingWindowCache"):
        class SlidingWindowCache:
            pass

        cache_utils.SlidingWindowCache = SlidingWindowCache


def load_local_qwen_modules(repo_root: Path):
    ensure_transformers_compat()

    package_name = "gated_attention_local"
    package = types.ModuleType(package_name)
    package.__path__ = [str(repo_root)]
    sys.modules[package_name] = package

    for module_name in ("configuration_qwen3", "modeling_qwen3"):
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.{module_name}", repo_root / f"{module_name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

    config_cls = sys.modules[f"{package_name}.configuration_qwen3"].Qwen3Config
    model_cls = sys.modules[f"{package_name}.modeling_qwen3"].Qwen3ForCausalLM
    return config_cls, model_cls


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TokenBlockDataset(Dataset):
    def __init__(self, blocks):
        if not blocks:
            raise ValueError("Dataset does not contain any complete token blocks.")
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        block = self.blocks[index]
        return {
            "input_ids": block.clone(),
            "attention_mask": torch.ones_like(block),
            "labels": block.clone(),
        }


def build_token_blocks(tokenizer, path: Path, block_size: int, lines_per_chunk: int = 1024):
    if not path.exists():
        raise FileNotFoundError(path)

    blocks = []
    buffer = []
    line_chunk = []

    def flush_lines():
        nonlocal buffer, line_chunk
        if not line_chunk:
            return
        text = "\n".join(line_chunk)
        line_chunk = []
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if not token_ids:
            return
        buffer.extend(token_ids)
        full_tokens = (len(buffer) // block_size) * block_size
        if full_tokens == 0:
            return
        tensor = torch.tensor(buffer[:full_tokens], dtype=torch.long).view(-1, block_size)
        blocks.extend(tensor)
        buffer = buffer[full_tokens:]

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            line_chunk.append(line)
            if len(line_chunk) >= lines_per_chunk:
                flush_lines()

    flush_lines()

    if buffer:
        if len(buffer) >= block_size:
            tensor = torch.tensor(buffer[: (len(buffer) // block_size) * block_size], dtype=torch.long).view(-1, block_size)
            blocks.extend(tensor)

    if not blocks:
        raise ValueError(f"Not enough tokens to form one block of length {block_size}: {path}")

    return blocks


def build_datasets(args, tokenizer):
    train_blocks = build_token_blocks(tokenizer, args.train_file, args.context_length)
    valid_blocks = build_token_blocks(tokenizer, args.valid_file, args.context_length)
    train_dataset = TokenBlockDataset(train_blocks)
    valid_dataset = TokenBlockDataset(valid_blocks)
    return train_dataset, valid_dataset


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step: int):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_special_token_id(primary, fallback):
    return primary if primary is not None else fallback


def build_config(args, tokenizer, config_cls, variant_name: str):
    pad_token_id = resolve_special_token_id(tokenizer.pad_token_id, tokenizer.eos_token_id)
    if pad_token_id is None:
        pad_token_id = 0
    eos_token_id = resolve_special_token_id(tokenizer.eos_token_id, pad_token_id)
    bos_token_id = resolve_special_token_id(tokenizer.bos_token_id, eos_token_id)

    rope_scaling = None
    if "default" not in ROPE_INIT_FUNCTIONS:
        rope_scaling = {"rope_type": "linear", "factor": 1.0}

    config_kwargs = dict(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        max_position_embeddings=max(args.context_length, args.max_position_embeddings or args.context_length),
        attention_bias=args.qkv_bias,
        qkv_bias=args.qkv_bias,
        attention_dropout=args.attention_dropout,
        use_qk_norm=args.use_qk_norm,
        rope_scaling=rope_scaling,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        use_cache=False,
        _attn_implementation="eager",
    )

    variant_kwargs = resolve_variant_config(variant_name, args.head_dim)
    config_kwargs.update(variant_kwargs)

    return config_cls(**config_kwargs)


def resolve_variant_config(variant_name: str, head_dim: int):
    if variant_name == "baseline":
        return {}

    if variant_name == "shared-headwise":
        return {"num_gate_groups": 1}

    if variant_name == "shared-elementwise":
        return {"num_gate_groups": head_dim}

    if variant_name == "a-headwise":
        return {
            "headwise_attn_output_gate": True,
            "independent_attn_output_gate": True,
        }

    hybrid_match = re.fullmatch(r"shared-hybrid-g(\d+)-lambda(\d+(?:p\d+)?)", variant_name)
    if hybrid_match:
        num_gate_groups = int(hybrid_match.group(1))
        if head_dim % num_gate_groups != 0:
            raise ValueError(
                f"Variant `{variant_name}` is invalid because head_dim ({head_dim}) "
                f"is not divisible by num_gate_groups ({num_gate_groups})."
            )
        hybrid_gate_lambda = float(hybrid_match.group(2).replace("p", "."))
        if not 0.0 <= hybrid_gate_lambda <= 1.0:
            raise ValueError(f"Variant `{variant_name}` has invalid lambda {hybrid_gate_lambda}.")
        return {
            "num_gate_groups": num_gate_groups,
            "hybrid_attn_output_gate": True,
            "hybrid_gate_lambda": hybrid_gate_lambda,
        }

    match = re.fullmatch(
        r"shared-groupwise-g(\d+)(?:-alpha(\d+(?:p\d+)?))?(?:-tau(\d+(?:p\d+)?))?",
        variant_name,
    )
    if match:
        num_gate_groups = int(match.group(1))
        if head_dim % num_gate_groups != 0:
            raise ValueError(
                f"Variant `{variant_name}` is invalid because head_dim ({head_dim}) "
                f"is not divisible by num_gate_groups ({num_gate_groups})."
            )
        config_kwargs = {"num_gate_groups": num_gate_groups}
        alpha_token = match.group(2)
        if alpha_token is not None:
            alpha = float(alpha_token.replace("p", "."))
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"Variant `{variant_name}` has invalid alpha {alpha}.")
            config_kwargs["attn_output_gate_residual_alpha"] = alpha
        tau_token = match.group(3)
        if tau_token is not None:
            temperature = float(tau_token.replace("p", "."))
            if temperature <= 0.0:
                raise ValueError(f"Variant `{variant_name}` has invalid temperature {temperature}.")
            config_kwargs["attn_output_gate_temperature"] = temperature
        return config_kwargs

    raise ValueError(f"Unsupported variant: {variant_name}")


def move_batch_to_device(batch, device: str):
    return {key: value.to(device) for key, value in batch.items()}


def mean_or_none(values):
    if not values:
        return None
    return sum(values) / len(values)


def average_layer_metrics(metric_batches):
    if not metric_batches:
        return []
    num_layers = len(metric_batches[0])
    return [
        sum(batch[layer_index] for batch in metric_batches) / len(metric_batches)
        for layer_index in range(num_layers)
    ]


def collect_gate_metrics(model):
    layer_means = []
    layer_stds = []
    layer_s01 = []
    layer_s02 = []
    layer_raw_means = []
    layer_raw_stds = []
    layer_raw_s01 = []
    layer_raw_s02 = []
    layer_base_means = []
    layer_base_stds = []
    layer_base_s01 = []
    layer_base_s02 = []
    gate_temperatures = []
    residual_alphas = []
    hybrid_gate_lambdas = []
    layer_head_gate_mean = []
    layer_head_gate_std = []
    layer_head_gate_sparsity_02 = []
    layer_group_gate_mean = []
    layer_group_gate_std = []
    layer_group_gate_sparsity_02 = []
    layer_hybrid_gate_mean = []
    layer_hybrid_gate_std = []
    layer_hybrid_gate_sparsity_02 = []
    layer_head_group_pearson = []
    layer_head_group_cosine = []
    for layer in model.model.layers:
        stats = getattr(layer.self_attn, "last_gate_stats", None)
        if not stats:
            continue
        layer_means.append(stats["mean"])
        layer_stds.append(stats.get("std", 0.0))
        layer_s01.append(stats["sparsity_01"])
        layer_s02.append(stats["sparsity_02"])
        layer_raw_means.append(stats.get("raw_mean", stats["mean"]))
        layer_raw_stds.append(stats.get("raw_std", stats.get("std", 0.0)))
        layer_raw_s01.append(stats.get("raw_sparsity_01", stats["sparsity_01"]))
        layer_raw_s02.append(stats.get("raw_sparsity_02", stats["sparsity_02"]))
        layer_base_means.append(stats.get("base_mean", stats.get("raw_mean", stats["mean"])))
        layer_base_stds.append(stats.get("base_std", stats.get("raw_std", stats.get("std", 0.0))))
        layer_base_s01.append(stats.get("base_sparsity_01", stats.get("raw_sparsity_01", stats["sparsity_01"])))
        layer_base_s02.append(stats.get("base_sparsity_02", stats.get("raw_sparsity_02", stats["sparsity_02"])))
        gate_temperatures.append(stats.get("temperature", 1.0))
        residual_alphas.append(stats.get("residual_alpha", 0.0))
        hybrid_gate_lambdas.append(stats.get("hybrid_gate_lambda"))
        if stats.get("head_gate_mean") is not None:
            layer_head_gate_mean.append(stats["head_gate_mean"])
            layer_head_gate_std.append(stats["head_gate_std"])
            layer_head_gate_sparsity_02.append(stats["head_gate_sparsity_02"])
            layer_group_gate_mean.append(stats["group_gate_mean"])
            layer_group_gate_std.append(stats["group_gate_std"])
            layer_group_gate_sparsity_02.append(stats["group_gate_sparsity_02"])
            layer_hybrid_gate_mean.append(stats["hybrid_gate_mean"])
            layer_hybrid_gate_std.append(stats["hybrid_gate_std"])
            layer_hybrid_gate_sparsity_02.append(stats["hybrid_gate_sparsity_02"])
            layer_head_group_pearson.append(stats["head_group_pearson"])
            layer_head_group_cosine.append(stats["head_group_cosine"])

    return {
        "gate_temperature": mean_or_none(gate_temperatures),
        "gate_residual_alpha": mean_or_none(residual_alphas),
        "hybrid_gate_lambda": mean_or_none([value for value in hybrid_gate_lambdas if value is not None]),
        "gate_mean": mean_or_none(layer_means),
        "gate_std": mean_or_none(layer_stds),
        "sparsity_01": mean_or_none(layer_s01),
        "sparsity_02": mean_or_none(layer_s02),
        "raw_gate_mean": mean_or_none(layer_raw_means),
        "raw_gate_std": mean_or_none(layer_raw_stds),
        "raw_sparsity_01": mean_or_none(layer_raw_s01),
        "raw_sparsity_02": mean_or_none(layer_raw_s02),
        "base_gate_mean": mean_or_none(layer_base_means),
        "base_gate_std": mean_or_none(layer_base_stds),
        "base_sparsity_01": mean_or_none(layer_base_s01),
        "base_sparsity_02": mean_or_none(layer_base_s02),
        "head_gate_mean": mean_or_none(layer_head_gate_mean),
        "head_gate_std": mean_or_none(layer_head_gate_std),
        "head_gate_sparsity_02": mean_or_none(layer_head_gate_sparsity_02),
        "group_gate_mean": mean_or_none(layer_group_gate_mean),
        "group_gate_std": mean_or_none(layer_group_gate_std),
        "group_gate_sparsity_02": mean_or_none(layer_group_gate_sparsity_02),
        "hybrid_gate_mean": mean_or_none(layer_hybrid_gate_mean),
        "hybrid_gate_std": mean_or_none(layer_hybrid_gate_std),
        "hybrid_gate_sparsity_02": mean_or_none(layer_hybrid_gate_sparsity_02),
        "head_group_pearson": mean_or_none(layer_head_group_pearson),
        "head_group_cosine": mean_or_none(layer_head_group_cosine),
        "layer_gate_mean": layer_means,
        "layer_gate_std": layer_stds,
        "layer_sparsity_01": layer_s01,
        "layer_sparsity_02": layer_s02,
        "layer_raw_gate_mean": layer_raw_means,
        "layer_raw_gate_std": layer_raw_stds,
        "layer_raw_sparsity_01": layer_raw_s01,
        "layer_raw_sparsity_02": layer_raw_s02,
        "layer_base_gate_mean": layer_base_means,
        "layer_base_gate_std": layer_base_stds,
        "layer_base_sparsity_01": layer_base_s01,
        "layer_base_sparsity_02": layer_base_s02,
        "layer_head_gate_mean": layer_head_gate_mean,
        "layer_head_gate_std": layer_head_gate_std,
        "layer_head_gate_sparsity_02": layer_head_gate_sparsity_02,
        "layer_group_gate_mean": layer_group_gate_mean,
        "layer_group_gate_std": layer_group_gate_std,
        "layer_group_gate_sparsity_02": layer_group_gate_sparsity_02,
        "layer_hybrid_gate_mean": layer_hybrid_gate_mean,
        "layer_hybrid_gate_std": layer_hybrid_gate_std,
        "layer_hybrid_gate_sparsity_02": layer_hybrid_gate_sparsity_02,
        "layer_head_group_pearson": layer_head_group_pearson,
        "layer_head_group_cosine": layer_head_group_cosine,
    }


def evaluate(model, valid_loader, device: str, max_eval_batches: int):
    model.eval()
    losses = []
    sink_all_values = []
    sink_excl_self_values = []
    gate_means = []
    gate_stds = []
    gate_s01 = []
    gate_s02 = []
    raw_gate_means = []
    raw_gate_stds = []
    raw_gate_s01 = []
    raw_gate_s02 = []
    base_gate_means = []
    base_gate_stds = []
    base_gate_s01 = []
    base_gate_s02 = []
    gate_temperatures = []
    gate_residual_alphas = []
    hybrid_gate_lambdas = []
    head_gate_means = []
    head_gate_stds = []
    head_gate_s02 = []
    group_gate_means = []
    group_gate_stds = []
    group_gate_s02 = []
    hybrid_gate_means = []
    hybrid_gate_stds = []
    hybrid_gate_s02 = []
    head_group_pearsons = []
    head_group_cosines = []
    layer_gate_mean_batches = []
    layer_gate_std_batches = []
    layer_s01_batches = []
    layer_s02_batches = []
    layer_raw_gate_mean_batches = []
    layer_raw_gate_std_batches = []
    layer_raw_s01_batches = []
    layer_raw_s02_batches = []
    layer_base_gate_mean_batches = []
    layer_base_gate_std_batches = []
    layer_base_s01_batches = []
    layer_base_s02_batches = []
    layer_head_gate_mean_batches = []
    layer_head_gate_std_batches = []
    layer_head_gate_s02_batches = []
    layer_group_gate_mean_batches = []
    layer_group_gate_std_batches = []
    layer_group_gate_s02_batches = []
    layer_hybrid_gate_mean_batches = []
    layer_hybrid_gate_std_batches = []
    layer_hybrid_gate_s02_batches = []
    layer_head_group_pearson_batches = []
    layer_head_group_cosine_batches = []

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
                output_attentions=True,
            )

            losses.append(outputs.loss.item())

            attentions = torch.stack(outputs.attentions)
            sink_all_values.append(attentions[..., 0].mean().item())
            if attentions.shape[-2] > 1:
                sink_excl_self_values.append(attentions[..., 1:, 0].mean().item())

            gate_metrics = collect_gate_metrics(model)
            if gate_metrics["gate_mean"] is not None:
                gate_temperatures.append(gate_metrics["gate_temperature"])
                gate_residual_alphas.append(gate_metrics["gate_residual_alpha"])
                if gate_metrics["hybrid_gate_lambda"] is not None:
                    hybrid_gate_lambdas.append(gate_metrics["hybrid_gate_lambda"])
                gate_means.append(gate_metrics["gate_mean"])
                gate_stds.append(gate_metrics["gate_std"])
                gate_s01.append(gate_metrics["sparsity_01"])
                gate_s02.append(gate_metrics["sparsity_02"])
                raw_gate_means.append(gate_metrics["raw_gate_mean"])
                raw_gate_stds.append(gate_metrics["raw_gate_std"])
                raw_gate_s01.append(gate_metrics["raw_sparsity_01"])
                raw_gate_s02.append(gate_metrics["raw_sparsity_02"])
                base_gate_means.append(gate_metrics["base_gate_mean"])
                base_gate_stds.append(gate_metrics["base_gate_std"])
                base_gate_s01.append(gate_metrics["base_sparsity_01"])
                base_gate_s02.append(gate_metrics["base_sparsity_02"])
                if gate_metrics["head_gate_mean"] is not None:
                    head_gate_means.append(gate_metrics["head_gate_mean"])
                    head_gate_stds.append(gate_metrics["head_gate_std"])
                    head_gate_s02.append(gate_metrics["head_gate_sparsity_02"])
                    group_gate_means.append(gate_metrics["group_gate_mean"])
                    group_gate_stds.append(gate_metrics["group_gate_std"])
                    group_gate_s02.append(gate_metrics["group_gate_sparsity_02"])
                    hybrid_gate_means.append(gate_metrics["hybrid_gate_mean"])
                    hybrid_gate_stds.append(gate_metrics["hybrid_gate_std"])
                    hybrid_gate_s02.append(gate_metrics["hybrid_gate_sparsity_02"])
                    head_group_pearsons.append(gate_metrics["head_group_pearson"])
                    head_group_cosines.append(gate_metrics["head_group_cosine"])
                layer_gate_mean_batches.append(gate_metrics["layer_gate_mean"])
                layer_gate_std_batches.append(gate_metrics["layer_gate_std"])
                layer_s01_batches.append(gate_metrics["layer_sparsity_01"])
                layer_s02_batches.append(gate_metrics["layer_sparsity_02"])
                layer_raw_gate_mean_batches.append(gate_metrics["layer_raw_gate_mean"])
                layer_raw_gate_std_batches.append(gate_metrics["layer_raw_gate_std"])
                layer_raw_s01_batches.append(gate_metrics["layer_raw_sparsity_01"])
                layer_raw_s02_batches.append(gate_metrics["layer_raw_sparsity_02"])
                layer_base_gate_mean_batches.append(gate_metrics["layer_base_gate_mean"])
                layer_base_gate_std_batches.append(gate_metrics["layer_base_gate_std"])
                layer_base_s01_batches.append(gate_metrics["layer_base_sparsity_01"])
                layer_base_s02_batches.append(gate_metrics["layer_base_sparsity_02"])
                if gate_metrics["layer_head_gate_mean"]:
                    layer_head_gate_mean_batches.append(gate_metrics["layer_head_gate_mean"])
                    layer_head_gate_std_batches.append(gate_metrics["layer_head_gate_std"])
                    layer_head_gate_s02_batches.append(gate_metrics["layer_head_gate_sparsity_02"])
                    layer_group_gate_mean_batches.append(gate_metrics["layer_group_gate_mean"])
                    layer_group_gate_std_batches.append(gate_metrics["layer_group_gate_std"])
                    layer_group_gate_s02_batches.append(gate_metrics["layer_group_gate_sparsity_02"])
                    layer_hybrid_gate_mean_batches.append(gate_metrics["layer_hybrid_gate_mean"])
                    layer_hybrid_gate_std_batches.append(gate_metrics["layer_hybrid_gate_std"])
                    layer_hybrid_gate_s02_batches.append(gate_metrics["layer_hybrid_gate_sparsity_02"])
                    layer_head_group_pearson_batches.append(gate_metrics["layer_head_group_pearson"])
                    layer_head_group_cosine_batches.append(gate_metrics["layer_head_group_cosine"])

    val_loss = mean_or_none(losses)
    ppl = None if val_loss is None else math.exp(min(val_loss, 20.0))
    metrics = {
        "val_loss": val_loss,
        "ppl": ppl,
        "sink_score_all": mean_or_none(sink_all_values),
        "sink_score_excl_self": mean_or_none(sink_excl_self_values),
        "gate_temperature": mean_or_none(gate_temperatures),
        "gate_residual_alpha": mean_or_none(gate_residual_alphas),
        "hybrid_gate_lambda": mean_or_none(hybrid_gate_lambdas),
        "gate_mean": mean_or_none(gate_means),
        "gate_std": mean_or_none(gate_stds),
        "sparsity_01": mean_or_none(gate_s01),
        "sparsity_02": mean_or_none(gate_s02),
        "raw_gate_mean": mean_or_none(raw_gate_means),
        "raw_gate_std": mean_or_none(raw_gate_stds),
        "raw_sparsity_01": mean_or_none(raw_gate_s01),
        "raw_sparsity_02": mean_or_none(raw_gate_s02),
        "base_gate_mean": mean_or_none(base_gate_means),
        "base_gate_std": mean_or_none(base_gate_stds),
        "base_sparsity_01": mean_or_none(base_gate_s01),
        "base_sparsity_02": mean_or_none(base_gate_s02),
        "head_gate_mean": mean_or_none(head_gate_means),
        "head_gate_std": mean_or_none(head_gate_stds),
        "head_gate_sparsity_02": mean_or_none(head_gate_s02),
        "group_gate_mean": mean_or_none(group_gate_means),
        "group_gate_std": mean_or_none(group_gate_stds),
        "group_gate_sparsity_02": mean_or_none(group_gate_s02),
        "hybrid_gate_mean": mean_or_none(hybrid_gate_means),
        "hybrid_gate_std": mean_or_none(hybrid_gate_stds),
        "hybrid_gate_sparsity_02": mean_or_none(hybrid_gate_s02),
        "head_group_pearson": mean_or_none(head_group_pearsons),
        "head_group_cosine": mean_or_none(head_group_cosines),
        "layer_gate_mean": average_layer_metrics(layer_gate_mean_batches),
        "layer_gate_std": average_layer_metrics(layer_gate_std_batches),
        "layer_sparsity_01": average_layer_metrics(layer_s01_batches),
        "layer_sparsity_02": average_layer_metrics(layer_s02_batches),
        "layer_raw_gate_mean": average_layer_metrics(layer_raw_gate_mean_batches),
        "layer_raw_gate_std": average_layer_metrics(layer_raw_gate_std_batches),
        "layer_raw_sparsity_01": average_layer_metrics(layer_raw_s01_batches),
        "layer_raw_sparsity_02": average_layer_metrics(layer_raw_s02_batches),
        "layer_base_gate_mean": average_layer_metrics(layer_base_gate_mean_batches),
        "layer_base_gate_std": average_layer_metrics(layer_base_gate_std_batches),
        "layer_base_sparsity_01": average_layer_metrics(layer_base_s01_batches),
        "layer_base_sparsity_02": average_layer_metrics(layer_base_s02_batches),
        "layer_head_gate_mean": average_layer_metrics(layer_head_gate_mean_batches),
        "layer_head_gate_std": average_layer_metrics(layer_head_gate_std_batches),
        "layer_head_gate_sparsity_02": average_layer_metrics(layer_head_gate_s02_batches),
        "layer_group_gate_mean": average_layer_metrics(layer_group_gate_mean_batches),
        "layer_group_gate_std": average_layer_metrics(layer_group_gate_std_batches),
        "layer_group_gate_sparsity_02": average_layer_metrics(layer_group_gate_s02_batches),
        "layer_hybrid_gate_mean": average_layer_metrics(layer_hybrid_gate_mean_batches),
        "layer_hybrid_gate_std": average_layer_metrics(layer_hybrid_gate_std_batches),
        "layer_hybrid_gate_sparsity_02": average_layer_metrics(layer_hybrid_gate_s02_batches),
        "layer_head_group_pearson": average_layer_metrics(layer_head_group_pearson_batches),
        "layer_head_group_cosine": average_layer_metrics(layer_head_group_cosine_batches),
    }
    return metrics


def train_variant(variant_name, args, tokenizer, config_cls, model_cls, train_dataset, valid_dataset):
    set_seed(args.seed)
    config = build_config(args, tokenizer, config_cls, variant_name)
    model = model_cls(config).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.warmup_steps, args.max_steps)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.num_workers,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    variant_dir = args.output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(variant_dir)

    train_iter = iter(train_loader)
    running_loss = 0.0
    running_steps = 0
    best_metrics = None
    metrics_log_path = variant_dir / "metrics.jsonl"
    interval_step_times = []
    model_param_count = sum(param.numel() for param in model.parameters())

    with metrics_log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, args.max_steps + 1):
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

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            interval_step_times.append(time.perf_counter() - step_start)

            running_loss += loss.item()
            running_steps += 1

            if step % args.eval_interval == 0 or step == args.max_steps:
                eval_metrics = evaluate(model, valid_loader, args.device, args.max_eval_batches)
                record = {
                    "variant": variant_name,
                    "step": step,
                    "train_loss": running_loss / max(1, running_steps),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "avg_step_time": mean_or_none(interval_step_times),
                    "model_param_count": model_param_count,
                    **eval_metrics,
                }
                print(json.dumps(record, ensure_ascii=False))
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()

                if best_metrics is None or record["val_loss"] < best_metrics["val_loss"]:
                    best_metrics = record

                running_loss = 0.0
                running_steps = 0
                interval_step_times = []

    summary_path = variant_dir / "summary.json"
    summary_path.write_text(json.dumps(best_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return best_metrics


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for variant_name in args.variants:
        resolve_variant_config(variant_name, args.head_dim)

    config_cls, model_cls = load_local_qwen_modules(repo_root)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    train_dataset, valid_dataset = build_datasets(args, tokenizer)

    run_metadata = {
        "seed": args.seed,
        "variants": args.variants,
        "attn_implementation": "eager",
        "device": args.device,
        "train_file": str(args.train_file),
        "valid_file": str(args.valid_file),
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_steps": args.max_steps,
        "eval_interval": args.eval_interval,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    all_summaries = {}
    for variant_name in args.variants:
        all_summaries[variant_name] = train_variant(
            variant_name, args, tokenizer, config_cls, model_cls, train_dataset, valid_dataset
        )

    comparison_path = args.output_dir / "comparison_summary.json"
    comparison_path.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
