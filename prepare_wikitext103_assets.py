import argparse
import hashlib
import json
import shutil
from pathlib import Path

from train_first_round_compare import build_token_blocks


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare WikiText-103 text assets and a shared SentencePiece tokenizer.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir-name", type=str, default="tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def hash_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_path(path: Path):
    digest = hashlib.sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(hash_file(file_path).encode("utf-8"))
    return digest.hexdigest()


def normalize_lines(raw_lines):
    normalized = []
    for raw_line in raw_lines:
        line = raw_line.strip()
        if line:
            normalized.append(line)
    return normalized


def write_split(path: Path, lines):
    path.write_text("\n".join(lines), encoding="utf-8")


def load_wikitext_splits():
    hf_error = None
    try:
        from datasets import load_dataset

        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        return {
            "source": "huggingface",
            "train": dataset["train"]["text"],
            "validation": dataset["validation"]["text"],
        }
    except Exception as exc:  # noqa: BLE001
        hf_error = exc

    try:
        from modelscope.msdatasets import MsDataset
    except ImportError as exc:
        raise RuntimeError(
            "Unable to load WikiText-103 from Hugging Face, and `modelscope` is not installed for fallback."
        ) from hf_error if hf_error is not None else exc

    split_texts = {}
    for split_name in ("train", "validation"):
        subset = MsDataset.load("modelscope/wikitext", subset_name="wikitext-103-v1", split=split_name)
        if hasattr(subset, "to_hf_dataset"):
            subset = subset.to_hf_dataset()

        if isinstance(subset, dict):
            if "text" in subset:
                split_texts[split_name] = subset["text"]
                continue
            first_value = next(iter(subset.values()))
            split_texts[split_name] = first_value["text"]
            continue

        split_texts[split_name] = subset["text"]

    return {
        "source": "modelscope",
        "train": split_texts["train"],
        "validation": split_texts["validation"],
        "huggingface_error": None if hf_error is None else repr(hf_error),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "train.txt"
    valid_path = args.output_dir / "valid.txt"
    tokenizer_dir = args.output_dir / args.tokenizer_dir_name
    metadata_path = args.output_dir / "asset_metadata.json"

    if not args.force and train_path.exists() and valid_path.exists() and tokenizer_dir.exists():
        print(json.dumps({"status": "exists", "output_dir": str(args.output_dir)}, ensure_ascii=False))
        return

    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("`sentencepiece` is required to train the tokenizer.") from exc

    try:
        from transformers import LlamaTokenizer
    except ImportError as exc:
        raise RuntimeError("`transformers` is required to save the tokenizer.") from exc

    dataset_splits = load_wikitext_splits()
    train_lines = normalize_lines(dataset_splits["train"])
    valid_lines = normalize_lines(dataset_splits["validation"])

    write_split(train_path, train_lines)
    write_split(valid_path, valid_lines)

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    spm_prefix = args.output_dir / "wikitext103_spm32k"
    spm.SentencePieceTrainer.Train(
        input=str(train_path),
        model_prefix=str(spm_prefix),
        model_type="bpe",
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )

    model_path = spm_prefix.with_suffix(".model")
    vocab_path = spm_prefix.with_suffix(".vocab")

    tokenizer = LlamaTokenizer(
        vocab_file=str(model_path),
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        add_bos_token=False,
        add_eos_token=False,
    )
    tokenizer.save_pretrained(tokenizer_dir)
    shutil.copy2(model_path, tokenizer_dir / "tokenizer.model")
    shutil.copy2(vocab_path, tokenizer_dir / "tokenizer.vocab")

    if len(tokenizer) != args.vocab_size:
        raise ValueError(f"Tokenizer vocab size mismatch: expected {args.vocab_size}, got {len(tokenizer)}.")
    if tokenizer.pad_token_id != 0 or tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2 or tokenizer.unk_token_id != 3:
        raise ValueError("Tokenizer special token ids do not match the required fixed specification.")

    train_blocks = build_token_blocks(tokenizer, train_path, args.context_length)
    valid_blocks = build_token_blocks(tokenizer, valid_path, args.context_length)
    if any(block.shape[-1] != args.context_length for block in train_blocks + valid_blocks):
        raise ValueError("Token block length validation failed.")

    metadata = {
        "dataset_name": "wikitext",
        "dataset_config_name": "wikitext-103-raw-v1",
        "dataset_source": dataset_splits["source"],
        "train_line_count": len(train_lines),
        "validation_line_count": len(valid_lines),
        "context_length": args.context_length,
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "train_file": str(train_path),
        "valid_file": str(valid_path),
        "tokenizer_dir": str(tokenizer_dir),
        "train_file_hash": hash_file(train_path),
        "valid_file_hash": hash_file(valid_path),
        "tokenizer_hash": hash_path(tokenizer_dir),
    }
    if dataset_splits.get("huggingface_error") is not None:
        metadata["huggingface_error"] = dataset_splits["huggingface_error"]
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
