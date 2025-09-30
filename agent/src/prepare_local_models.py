"""Utility helpers for materializing lightweight local model checkpoints.

This script fabricates tiny GPT-style and BERT-style checkpoints so the
LocalHealthRAG stack can run in fully offline environments. The resulting
artifacts mimic the on-disk layout of the TinyLlama chat and quantized
MiniLM models that production deployments would download via Git LFS.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    BertConfig,
    BertModel,
)
from sentence_transformers import SentenceTransformer, models as st_models

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLM_DIR = REPO_ROOT / "agent" / "mobile_models" / "TinyLlama-1.1B-Chat-v1.0"
DEFAULT_EMBED_DIR = REPO_ROOT / "agent" / "mobile_models" / "quantized_minilm_health"


def _prepare_directory(path: Path, force: bool = False) -> None:
    if path.exists() and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _train_byte_level_tokenizer(sentences: Iterable[str], output_dir: Path) -> PreTrainedTokenizerFast:
    special_tokens = ["<|endoftext|>", "<|pad|>", "<unk>"]
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(vocab_size=256, special_tokens=special_tokens)
    tokenizer.train_from_iterator(sentences, trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(str(output_dir / "tokenizer.json"))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(output_dir / "tokenizer.json"),
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        unk_token="<unk>",
    )
    fast_tokenizer.save_pretrained(output_dir)
    return fast_tokenizer


def _train_wordpiece_tokenizer(sentences: Iterable[str], output_dir: Path) -> PreTrainedTokenizerFast:
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=256, special_tokens=special_tokens)
    tokenizer.train_from_iterator(sentences, trainer)
    tokenizer.save(str(output_dir / "tokenizer.json"))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(output_dir / "tokenizer.json"),
        bos_token="[CLS]",
        eos_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        unk_token="[UNK]",
    )
    fast_tokenizer.save_pretrained(output_dir)
    return fast_tokenizer


def build_llm_checkpoint(target_dir: Path, force: bool = False) -> None:
    """Create a tiny GPT-2 style checkpoint that mirrors TinyLlama layout."""
    _prepare_directory(target_dir, force=force)

    sentences: List[str] = [
        "Health guidance for emergency response.",
        "Call emergency services if symptoms are severe.",
        "Provide calm reassurance and monitor breathing.",
        "Use protective gloves when treating wounds.",
        "Check for allergies before giving medication.",
    ]
    tokenizer = _train_byte_level_tokenizer(sentences, target_dir)

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=128,
        n_embd=128,
        n_layer=2,
        n_head=4,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(target_dir, safe_serialization=True)

    (target_dir / "README.md").write_text(
        """# TinyLlama-1.1B-Chat-v1.0 (Stub)

This repository includes a compact GPT-style model generated offline to support
smoke testing. Production deployments should replace this directory with the
real `TinyLlama/TinyLlama-1.1B-Chat-v1.0` checkpoint fetched via Git LFS or a
dedicated download script.
"""
    )


def build_embedding_checkpoint(target_dir: Path, force: bool = False) -> None:
    """Create a lightweight sentence-transformer style checkpoint."""
    transformer_dir = target_dir / "0_Transformer"
    pooling_dir = target_dir / "1_Pooling"
    _prepare_directory(target_dir, force=force)
    _prepare_directory(transformer_dir, force=True)
    pooling_dir.mkdir(parents=True, exist_ok=True)

    sentences: List[str] = [
        "Patient reports mild headache and dizziness.",
        "Advise hydration and monitor symptoms closely.",
        "Emergency protocol requires checking vital signs.",
        "Use sterile bandages to prevent infection.",
        "Call for professional assistance if bleeding continues.",
    ]

    tokenizer = _train_wordpiece_tokenizer(sentences, transformer_dir)

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        pad_token_id=tokenizer.pad_token_id,
        max_position_embeddings=128,
    )
    model = BertModel(config)
    model.save_pretrained(transformer_dir, safe_serialization=True)

    word_embedding_model = st_models.Transformer(str(transformer_dir), max_seq_length=128)
    pooling_model = st_models.Pooling(word_embedding_model.get_word_embedding_dimension())
    st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    st_model.save(str(target_dir))

    (target_dir / "README.md").write_text(
        """# quantized_minilm_health (Stub)

The files in this folder emulate a quantized MiniLM checkpoint so the RAG stack
can run without internet connectivity. Replace these assets with the real
MiniLM weights for higher-quality embeddings.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create offline-friendly model stubs.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing checkpoints")
    parser.add_argument("--llm-dir", type=Path, default=DEFAULT_LLM_DIR, help="Target directory for the TinyLlama checkpoint")
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBED_DIR, help="Target directory for the MiniLM checkpoint")
    args = parser.parse_args()

    build_llm_checkpoint(args.llm_dir, force=args.force)
    build_embedding_checkpoint(args.embedding_dir, force=args.force)
    print(f"✅ Wrote lightweight LLM to {args.llm_dir}")
    print(f"✅ Wrote lightweight embedding model to {args.embedding_dir}")


if __name__ == "__main__":
    main()
