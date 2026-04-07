import copy
import json
from pathlib import Path


SCHEMA_VERSION = 1
DATASET_NAME = "wikitext"
DATASET_CONFIG_NAME = "wikitext-103-raw-v1"
TOKENIZER_NAME = "wikitext103_spm32k"
RESULT_ROOT_NAME = "wikitext103_spm32k_12L512d_ctx256"
FIXED_SEEDS = [13, 42, 3407]

FIXED_MODEL_CONFIG = {
    "num_hidden_layers": 12,
    "hidden_size": 512,
    "intermediate_size": 1536,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "context_length": 256,
    "attention_dropout": 0.0,
    "qkv_bias": False,
    "use_qk_norm": False,
}

FIXED_TRAINING_CONFIG = {
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_steps": 300,
    "batch_size": 4,
    "eval_batch_size": 4,
    "max_steps": 3000,
    "eval_interval": 250,
    "max_eval_batches": 0,
    "max_grad_norm": 1.0,
    "num_workers": 0,
}


def gate_none():
    return {
        "gate_type": "none",
        "num_gate_groups": None,
        "basis_rank": None,
        "basis_alpha_norm": False,
        "basis_temperature": 1.0,
    }


def gate_headwise():
    return {
        "gate_type": "shared_headwise",
        "num_gate_groups": None,
        "basis_rank": None,
        "basis_alpha_norm": False,
        "basis_temperature": 1.0,
    }


def gate_groupwise(num_gate_groups):
    return {
        "gate_type": "shared_groupwise",
        "num_gate_groups": num_gate_groups,
        "basis_rank": None,
        "basis_alpha_norm": False,
        "basis_temperature": 1.0,
    }


def gate_elementwise():
    return {
        "gate_type": "shared_elementwise",
        "num_gate_groups": None,
        "basis_rank": None,
        "basis_alpha_norm": False,
        "basis_temperature": 1.0,
    }


def gate_basis(rank, alpha_norm=False, temperature=1.0):
    return {
        "gate_type": "basis",
        "num_gate_groups": None,
        "basis_rank": rank,
        "basis_alpha_norm": alpha_norm,
        "basis_temperature": temperature,
    }


TOKEN_TO_SPEC = {
    "N": gate_none,
    "H": gate_headwise,
    "E": gate_elementwise,
    "G2": lambda: gate_groupwise(2),
    "G4": lambda: gate_groupwise(4),
    "G8": lambda: gate_groupwise(8),
    "G16": lambda: gate_groupwise(16),
    "G32": lambda: gate_groupwise(32),
}


def repeat_spec(spec_factory, depth=FIXED_MODEL_CONFIG["num_hidden_layers"]):
    return [copy.deepcopy(spec_factory()) for _ in range(depth)]


def layout_from_tokens(tokens, basis_rank=None, basis_alpha_norm=False, basis_temperature=1.0):
    layout = []
    for token in tokens:
        if token == "B":
            if basis_rank is None:
                raise ValueError("`basis_rank` is required when layout tokens contain `B`.")
            layout.append(gate_basis(basis_rank, alpha_norm=basis_alpha_norm, temperature=basis_temperature))
        else:
            try:
                layout.append(copy.deepcopy(TOKEN_TO_SPEC[token]()))
            except KeyError as exc:
                raise ValueError(f"Unsupported layout token: {token}") from exc
    return layout


def build_manifest(suite_name, variant_id, layer_gate_layout, references):
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": DATASET_NAME,
        "dataset_config_name": DATASET_CONFIG_NAME,
        "tokenizer_name": TOKENIZER_NAME,
        "suite_name": suite_name,
        "variant_id": variant_id,
        "references": list(references),
        "seeds": list(FIXED_SEEDS),
        "model_config": copy.deepcopy(FIXED_MODEL_CONFIG),
        "training_config": copy.deepcopy(FIXED_TRAINING_CONFIG),
        "layer_gate_layout": copy.deepcopy(layer_gate_layout),
    }


def build_baseline_manifests():
    return [
        build_manifest("baseline", "uniform-baseline", repeat_spec(gate_none), []),
        build_manifest("baseline", "uniform-headwise", repeat_spec(gate_headwise), ["uniform-baseline"]),
        build_manifest("baseline", "uniform-groupwise-g2", repeat_spec(lambda: gate_groupwise(2)), ["uniform-baseline"]),
        build_manifest("baseline", "uniform-groupwise-g4", repeat_spec(lambda: gate_groupwise(4)), ["uniform-baseline"]),
        build_manifest("baseline", "uniform-groupwise-g8", repeat_spec(lambda: gate_groupwise(8)), ["uniform-baseline"]),
        build_manifest("baseline", "uniform-groupwise-g16", repeat_spec(lambda: gate_groupwise(16)), ["uniform-baseline"]),
        build_manifest("baseline", "uniform-groupwise-g32", repeat_spec(lambda: gate_groupwise(32)), ["uniform-baseline"]),
        build_manifest("baseline", "uniform-elementwise", repeat_spec(gate_elementwise), ["uniform-baseline"]),
    ]


def build_depth_manifests():
    return [
        build_manifest("depth_adaptive", "depth-E4-H4-N4", layout_from_tokens(["E"] * 4 + ["H"] * 4 + ["N"] * 4), ["depth-N4-H4-E4", "depth-random-E4-H4-N4", "uniform-elementwise"]),
        build_manifest("depth_adaptive", "depth-G8x4-H4-N4", layout_from_tokens(["G8"] * 4 + ["H"] * 4 + ["N"] * 4), ["depth-N4-H4-G8x4", "depth-random-G8x4-H4-N4", "uniform-groupwise-g8"]),
        build_manifest("depth_adaptive", "depth-E6-H6", layout_from_tokens(["E"] * 6 + ["H"] * 6), ["depth-random-E6-H6", "uniform-elementwise"]),
        build_manifest("depth_adaptive", "depth-G8x6-H6", layout_from_tokens(["G8"] * 6 + ["H"] * 6), ["depth-random-G8x6-H6", "uniform-groupwise-g8"]),
        build_manifest("depth_adaptive", "depth-N4-H4-E4", layout_from_tokens(["N"] * 4 + ["H"] * 4 + ["E"] * 4), []),
        build_manifest("depth_adaptive", "depth-N4-H4-G8x4", layout_from_tokens(["N"] * 4 + ["H"] * 4 + ["G8"] * 4), []),
        build_manifest("depth_adaptive", "depth-E3-G8x3-H3-N3", layout_from_tokens(["E"] * 3 + ["G8"] * 3 + ["H"] * 3 + ["N"] * 3), []),
        build_manifest("depth_adaptive", "depth-N3-H3-G8x3-E3", layout_from_tokens(["N"] * 3 + ["H"] * 3 + ["G8"] * 3 + ["E"] * 3), []),
        build_manifest("depth_adaptive", "depth-random-E4-H4-N4", layout_from_tokens(["H", "N", "E", "H", "E", "N", "H", "E", "N", "H", "E", "N"]), []),
        build_manifest("depth_adaptive", "depth-random-G8x4-H4-N4", layout_from_tokens(["N", "G8", "H", "G8", "N", "H", "H", "N", "G8", "N", "H", "G8"]), []),
        build_manifest("depth_adaptive", "depth-random-E6-H6", layout_from_tokens(["E", "H", "H", "E", "H", "E", "E", "H", "E", "H", "H", "E"]), []),
        build_manifest("depth_adaptive", "depth-random-G8x6-H6", layout_from_tokens(["H", "G8", "H", "G8", "G8", "H", "H", "G8", "H", "G8", "G8", "H"]), []),
    ]


def build_basis_rank_manifests():
    manifests = []
    for rank in (4, 8, 16):
        manifests.append(
            build_manifest(
                "basis_rank",
                f"basis-r{rank}",
                repeat_spec(lambda rank=rank: gate_basis(rank)),
                ["uniform-headwise", "uniform-groupwise-g8", "uniform-elementwise"],
            )
        )
    return manifests


def build_basis_stability_manifests(best_basis_rank):
    return [
        build_manifest(
            "basis_stability",
            f"basis-r{best_basis_rank}-l2norm",
            repeat_spec(lambda: gate_basis(best_basis_rank, alpha_norm=True)),
            [f"basis-r{best_basis_rank}"],
        ),
        build_manifest(
            "basis_stability",
            f"basis-r{best_basis_rank}-tau0p7",
            repeat_spec(lambda: gate_basis(best_basis_rank, temperature=0.7)),
            [f"basis-r{best_basis_rank}"],
        ),
        build_manifest(
            "basis_stability",
            f"basis-r{best_basis_rank}-tau1p3",
            repeat_spec(lambda: gate_basis(best_basis_rank, temperature=1.3)),
            [f"basis-r{best_basis_rank}"],
        ),
    ]


def build_combination_manifests(best_basis_rank):
    return [
        build_manifest(
            "combination",
            "combo-basis4-H8",
            layout_from_tokens(["B"] * 4 + ["H"] * 8, basis_rank=best_basis_rank),
            [f"basis-r{best_basis_rank}", "uniform-headwise", "depth-E4-H4-N4", "depth-G8x4-H4-N4"],
        ),
        build_manifest(
            "combination",
            "combo-basis4-H4-N4",
            layout_from_tokens(["B"] * 4 + ["H"] * 4 + ["N"] * 4, basis_rank=best_basis_rank),
            [f"basis-r{best_basis_rank}", "uniform-headwise", "depth-E4-H4-N4", "depth-G8x4-H4-N4"],
        ),
        build_manifest(
            "combination",
            "combo-basis6-H6",
            layout_from_tokens(["B"] * 6 + ["H"] * 6, basis_rank=best_basis_rank),
            [f"basis-r{best_basis_rank}", "uniform-headwise", "depth-E4-H4-N4", "depth-G8x4-H4-N4"],
        ),
        build_manifest(
            "combination",
            "combo-H8-basis4",
            layout_from_tokens(["H"] * 8 + ["B"] * 4, basis_rank=best_basis_rank),
            [f"basis-r{best_basis_rank}", "uniform-headwise", "depth-E4-H4-N4", "depth-G8x4-H4-N4"],
        ),
        build_manifest(
            "combination",
            "combo-basis4-G8x4-N4",
            layout_from_tokens(["B"] * 4 + ["G8"] * 4 + ["N"] * 4, basis_rank=best_basis_rank),
            [f"basis-r{best_basis_rank}", "uniform-headwise", "depth-E4-H4-N4", "depth-G8x4-H4-N4"],
        ),
    ]


def build_phase1_manifests():
    return build_baseline_manifests() + build_depth_manifests() + build_basis_rank_manifests()


def build_phase2_manifests(best_basis_rank):
    return build_basis_stability_manifests(best_basis_rank) + build_combination_manifests(best_basis_rank)


def build_full_manifests(best_basis_rank):
    return build_phase1_manifests() + build_phase2_manifests(best_basis_rank)


def manifest_index(manifests):
    return {manifest["variant_id"]: manifest for manifest in manifests}


def select_best_basis_rank(summary_records):
    summary_by_variant = {record["variant_id"]: record for record in summary_records}
    candidates = []
    for rank in (4, 8, 16):
        variant_id = f"basis-r{rank}"
        if variant_id not in summary_by_variant:
            raise KeyError(f"Missing aggregated record for {variant_id}.")
        record = summary_by_variant[variant_id]
        candidates.append(
            (
                record["val_loss_mean"],
                record["extra_param_count"],
                record["extra_flops_per_token"],
                rank,
            )
        )
    candidates.sort()
    return candidates[0][-1]


def write_manifest(manifest, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def write_manifests(manifests, manifest_dir):
    manifest_dir = Path(manifest_dir)
    paths = {}
    for manifest in manifests:
        path = manifest_dir / f"{manifest['variant_id']}.json"
        write_manifest(manifest, path)
        paths[manifest["variant_id"]] = path
    return paths
