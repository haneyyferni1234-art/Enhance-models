"""
FusionModelWrapper
~~~~~~~~~~~~~~~~~
Lossless token fusion middleware for any HuggingFace CausalLM.

Compresses token sequences by ~11-17% using bigram + multi-gram phrase
detection, delivering O(N^2) prefill speedup of ~1.27-1.46x.

Speedup applies ONLY to the prefill (input encoding) phase, NOT to
autoregressive decoding.

NEW TOKENS USE RANDOM INITIALIZATION → fine-tuning required for quality.

Usage:
    model, tokenizer = FusionModelWrapper.from_pretrained("model-path")
    model.generate("your prompt")  # auto-fused
"""

import json, re, copy, math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


# ============================================================
# 1. TRIE FOR GREEDY LONGEST-MATCH TOKEN FUSION
# ============================================================

class FusionTrie:
    """Trie of base-token sequences mapping to fused token IDs."""

    def __init__(self):
        self.root = {}
        self.size = 0

    def insert(self, token_ids, fused_id):
        node = self.root
        for tid in token_ids:
            if tid not in node:
                node[tid] = {}
            node = node[tid]
        if "fid" not in node:
            node["fid"] = fused_id
            self.size += 1

    def longest_match(self, token_ids, start):
        node = self.root
        best_len = 0
        best_fid = None
        j = start
        while j < len(token_ids) and token_ids[j] in node:
            node = node[token_ids[j]]
            j += 1
            if "fid" in node:
                best_len = j - start
                best_fid = node["fid"]
        return best_len, best_fid

    def fuse(self, token_ids):
        out = []
        i = 0
        while i < len(token_ids):
            length, fid = self.longest_match(token_ids, i)
            if length > 0:
                out.append(fid)
                i += length
            else:
                out.append(token_ids[i])
                i += 1
        return out

    def __len__(self):
        return self.size


# ============================================================
# 2. PATTERN LOADER
# ============================================================

def load_patterns(bigram_path, multigram_path, base_tokenizer, base_vocab_size=None):
    """Load and merge bigram + multi-gram patterns into a single Trie.

    - Bigram patterns: text-level word pairs → encoded with base_tokenizer at load time
      (tokenizer-agnostic, works with any model).
    - Multi-gram patterns: token-level ID sequences mined with the specific model's
      tokenizer (model-specific, exact match).

    Args:
        base_vocab_size: Use model.config.vocab_size (not tokenizer.vocab_size)
                         to avoid overlapping with special tokens.
    """
    if base_vocab_size is None:
        base_vocab_size = base_tokenizer.vocab_size

    trie = FusionTrie()

    # ── Bigram patterns (store text, encode at load time) ──────────────
    bg_count = 0
    with open(bigram_path) as f:
        bg_data = json.load(f)

    for i, entry in enumerate(bg_data):
        w1, w2 = entry["words"]
        t1 = base_tokenizer.encode(" " + w1)
        t2 = base_tokenizer.encode(" " + w2)
        if len(t1) > 0 and len(t2) > 0:
            phrase = [t1[0], t2[0]]
            fid = base_vocab_size + i
            trie.insert(phrase, fid)
            bg_count += 1
    print(f"  [Fusion] Loaded {bg_count:,} bigram patterns")

    # ── Multi-gram patterns (stored as token IDs, no re-encoding) ─────
    mg_count = 0
    with open(multigram_path) as f:
        mg_data = json.load(f)
    for i, entry in enumerate(mg_data):
        phrase = entry["phrase"]
        fid = base_vocab_size + bg_count + i
        trie.insert(phrase, fid)
        mg_count += 1
    print(f"  [Fusion] Loaded {mg_count:,} multi-gram patterns")

    return trie


# ============================================================
# 3. EMBEDDING INITIALIZATION
# ============================================================

def random_init_embeddings(model, tokenizer, trie, base_vocab_size):
    """Initialize new embedding vectors with random noise.

    WARNING: Mean initialization (avg of constituent tokens) was tried but is
    theoretically unsound — mean("New", "York") does NOT represent "New York".
    The model has never seen these vectors; logits are uncalibrated.

    Training (LoRA) is REQUIRED for usable quality. This gives the model
    a random starting point to learn from.
    """
    embed = model.get_input_embeddings()
    new_size = embed.weight.shape[0]
    n_new = new_size - base_vocab_size

    if n_new <= 0:
        return

    print(f"  [Fusion] Initializing {n_new:,} new embeddings with random noise")
    print(f"  [Fusion] ⚠ WARNING: Zero-shot fusion is NOT production-safe without fine-tuning")

    device = embed.weight.device
    embed_weight = embed.weight.data

    # Sample random vectors from the existing embedding distribution
    # (mean ± 0.5 * std) for better initialization than pure N(0,1)
    existing = embed_weight[:base_vocab_size]
    mean = existing.mean(dim=0, keepdim=True)
    std = existing.std(dim=0, keepdim=True)

    noise = torch.randn(n_new, existing.shape[1], device=device, dtype=existing.dtype)
    embed_weight[base_vocab_size:] = mean + noise * std


# ============================================================
# 4. FUSION MODEL WRAPPER
# ============================================================

class FusionModelWrapper:
    """
    Wraps a HuggingFace CausalLM with lossless token fusion.

    ╔══════════════════════════════════════════════════════════════╗
    ║  SPEEDUP APPLIES ONLY TO PREFILL (input encoding).         ║
    ║  Autoregressive decoding is unaffected.                     ║
    ║                                                              ║
    ║  Fine-tuning (LoRA) on fused text REQUIRED for quality.     ║
    ╚══════════════════════════════════════════════════════════════╝

    Args:
        base_model: Preloaded HuggingFace model
        base_tokenizer: Corresponding tokenizer
        trie: FusionTrie instance with all patterns
        base_vocab_size: Original vocabulary size before expansion
    """

    def __init__(self, base_model, base_tokenizer, trie, base_vocab_size):
        self.model = base_model
        self.tokenizer = base_tokenizer
        self.trie = trie
        self.base_vocab_size = base_vocab_size
        self._device = base_model.device

        self._fid_to_constituents = {}
        self._build_reverse_map()

    def _build_reverse_map(self):
        def _walk(node, path):
            if "fid" in node:
                self._fid_to_constituents[node["fid"]] = path[:]
            for tid, child in node.items():
                if tid != "fid":
                    _walk(child, path + [tid])
        _walk(self.trie.root, [])

    # ---- Encoding / Decoding ----

    def encode(self, text_or_ids):
        """Fuse token sequence: shorter but semantically identical."""
        if isinstance(text_or_ids, str):
            ids = self.tokenizer.encode(text_or_ids)
        else:
            ids = text_or_ids
        fused = self.trie.fuse(ids)
        return fused

    def decode(self, ids, skip_special_tokens=True):
        """Expand fused tokens back to original token IDs then decode."""
        expanded = []
        for tid in ids:
            if tid in self._fid_to_constituents:
                expanded.extend(self._fid_to_constituents[tid])
            else:
                expanded.append(tid)
        return self.tokenizer.decode(expanded, skip_special_tokens=skip_special_tokens)

    def encode_for_model(self, text):
        fused = self.encode(text)
        ids = torch.tensor([fused], device=self.device, dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.long)
        return ids, mask

    # ---- Generation ----

    @torch.no_grad()
    def generate(self, prompt, **kwargs):
        """
        Generate with transparent fusion:
        1. Fuse input prompt (prefill speedup ~1.27-1.46x)
        2. Run model.generate on fused tokens
        3. Expand output back to original token space
        4. Decode
        """
        was_string = isinstance(prompt, str)
        if was_string:
            input_ids, attention_mask = self.encode_for_model(prompt)
        else:
            input_ids = prompt.to(self.device) if hasattr(prompt, 'device') else prompt
            attention_mask = kwargs.pop('attention_mask', None)

        max_new = kwargs.pop('max_new_tokens', 256)
        do_sample = kwargs.pop('do_sample', False)
        temperature = kwargs.pop('temperature', 0.0)

        out = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            **kwargs
        )

        if was_string:
            return self.decode(out[0].tolist())
        return out

    def generate_batch(self, prompts, max_new_tokens=256, temperature=0.0, do_sample=False):
        all_fused = [self.encode(p) for p in prompts]
        max_len = max(len(f) for f in all_fused)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        batched = []
        masks = []
        for fused in all_fused:
            pad_count = max_len - len(fused)
            batched.append(fused + [pad_id] * pad_count)
            masks.append([1] * len(fused) + [0] * pad_count)

        input_ids = torch.tensor(batched, device=self.device, dtype=torch.long)
        attention_mask = torch.tensor(masks, device=self.device, dtype=torch.long)

        out = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=pad_id,
        )

        decoded = []
        for seq in out:
            decoded.append(self.decode(seq.tolist()))
        return decoded

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # ---- Save / Load ----

    def save_pretrained(self, save_directory):
        import os, pickle
        os.makedirs(save_directory, exist_ok=True)

        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

        trie_path = os.path.join(save_directory, "fusion_trie.pkl")
        with open(trie_path, "wb") as f:
            pickle.dump(self.trie, f)

        meta = {
            "base_vocab_size": self.base_vocab_size,
            "new_vocab_size": self.model.config.vocab_size,
            "n_fusion_tokens": len(self.trie),
        }
        meta_path = os.path.join(save_directory, "fusion_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        print(f"  [Fusion] Saved fused model to {save_directory}")
        print(f"  [Fusion]   Vocab: {meta['base_vocab_size']:,} + {meta['n_fusion_tokens']:,} fusion tokens")
        return save_directory

    # ---- Convenience ----

    def to(self, device):
        self.model = self.model.to(device)
        self._device = device
        return self

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value


# ============================================================
# 5. FACTORY: from_pretrained
# ============================================================

def _resolve_pattern_path(filename):
    """Search for pattern files in patterns/ subdir, root dir, or alongside this script."""
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "patterns", filename),
        os.path.join(script_dir, filename),
        os.path.join(os.getcwd(), "patterns", filename),
        os.path.join(os.getcwd(), filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]  # fall back to patterns/ subdir

def from_pretrained(
    model_name_or_path,
    bigram_path=None,
    multigram_path=None,
    device_map="auto",
    torch_dtype=torch.float16,
    **model_kwargs
):
    """
    Load a model and wrap it with the Fusion token compressor.

    Speedup: ~1.27-1.46x on prefill (input encoding) only.
    Quality: Requires LoRA fine-tuning for production use.

    Returns:
        (FusionModelWrapper, tokenizer)
    """
    if bigram_path is None:
        bigram_path = _resolve_pattern_path("bigram_patterns.json")
    if multigram_path is None:
        multigram_path = _resolve_pattern_path("multigram_patterns.json")

    print("=" * 60)
    print("  FusionModelWrapper — lossless token fusion (prefill-only)")
    print("=" * 60)

    # 1. Load tokenizer + base model
    print("\n[1/4] Loading model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        **model_kwargs
    )
    base_vocab_size = model.config.vocab_size
    print(f"       Base vocab: {base_vocab_size:,}")

    # 2. Build fusion trie
    print("\n[2/4] Building fusion trie from patterns...")
    trie = load_patterns(bigram_path, multigram_path, tokenizer, base_vocab_size)
    print(f"       Trie size: {len(trie):,} patterns")

    # 3. Expand model vocabulary
    n_new = len(trie)
    new_vocab_size = base_vocab_size + n_new
    print(f"\n[3/4] Expanding vocabulary: {base_vocab_size:,} → {new_vocab_size:,}")
    print(f"       New tokens: {n_new:,} (+{n_new/base_vocab_size*100:.2f}%)")

    model.resize_token_embeddings(new_vocab_size)
    random_init_embeddings(model, tokenizer, trie, base_vocab_size)

    if not model.config.tie_word_embeddings:
        lm_head = model.get_output_embeddings()
        if lm_head is not None:
            old_w = lm_head.weight.data
            new_w = torch.zeros(new_vocab_size, old_w.shape[1], dtype=old_w.dtype, device=old_w.device)
            new_w[:base_vocab_size] = old_w[:base_vocab_size]
            new_w[base_vocab_size:] = old_w[:base_vocab_size].mean(dim=0, keepdim=True)
            lm_head.weight = nn.Parameter(new_w)
            lm_head.out_features = new_vocab_size

    # 4. Build wrapper
    print("\n[4/4] Building wrapper...")
    print("\n  ⚠ NOTE: Zero-shot fusion is NOT calibrated for production.")
    print("  Run LoRA fine-tuning on fused text for proper quality.")
    wrapper = FusionModelWrapper(model, tokenizer, trie, base_vocab_size)

    return wrapper, tokenizer


# ============================================================
# 6. FROM_FUSED_PRETRAINED: LOAD A SAVED FUSED MODEL
# ============================================================

def from_fused_pretrained(save_directory, device_map="auto", torch_dtype=torch.float16, **model_kwargs):
    import os, pickle

    print("=" * 60)
    print("  FusionModelWrapper — loading fused checkpoint")
    print("=" * 60)

    meta_path = os.path.join(save_directory, "fusion_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"\n  Base vocab: {meta['base_vocab_size']:,}")
    print(f"  Fusion tokens: {meta['n_fusion_tokens']:,}")

    print("\n[1/3] Loading model & tokenizer from checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(save_directory, **model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        save_directory, device_map=device_map, torch_dtype=torch_dtype, **model_kwargs
    )
    print(f"  Vocab size: {model.config.vocab_size:,}")

    print("\n[2/3] Loading fusion trie...")
    trie_path = os.path.join(save_directory, "fusion_trie.pkl")
    with open(trie_path, "rb") as f:
        trie = pickle.load(f)
    print(f"  Trie size: {len(trie):,} patterns")

    print("\n[3/3] Building wrapper...")
    wrapper = FusionModelWrapper(model, tokenizer, trie, meta["base_vocab_size"])
    return wrapper, tokenizer


# ============================================================
# 7. CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FusionModelWrapper — lossless token fusion")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model path or name")
    parser.add_argument("--prompt", default="Explain how transformer self-attention works in machine learning models.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--save", default=None, help="Save fused model to this directory")
    parser.add_argument("--load", default=None, help="Load previously saved fused model from this directory")
    parser.add_argument("--benchmark", action="store_true", help="Run token compression benchmark on corpus")
    parser.add_argument("--batch", nargs="*", default=[], help="Batch of prompts for batch generation demo")
    parser.add_argument("--bigram-path", default=None, help="Bigram pattern file")
    parser.add_argument("--multigram-path", default=None, help="Multi-gram pattern file")
    args = parser.parse_args()

    kw = {}
    if args.bigram_path:
        kw["bigram_path"] = args.bigram_path
    if args.multigram_path:
        kw["multigram_path"] = args.multigram_path

    if args.load:
        wrapper, tokenizer = from_fused_pretrained(args.load, device_map="cpu", torch_dtype=torch.float32)
    else:
        wrapper, tokenizer = from_pretrained(args.model, device_map="cpu", torch_dtype=torch.float32, **kw)

    if args.save:
        wrapper.save_pretrained(args.save)

    if args.prompt:
        orig_ids = tokenizer.encode(args.prompt)
        fused_ids = wrapper.encode(args.prompt)
        print(f"\n  Prompt: {args.prompt[:80]}...")
        print(f"  Original tokens: {len(orig_ids)}")
        print(f"  Fused tokens:    {len(fused_ids)}")
        print(f"  Compression:     {(1-len(fused_ids)/len(orig_ids))*100:.1f}%")
        print(f"  Prefill speedup: {(len(orig_ids)**2)/(len(fused_ids)**2):.2f}x (O(N²) encoding only)")

    if args.batch:
        print(f"\n  Batch generation ({len(args.batch)} prompts):")
        outputs = wrapper.generate_batch(args.batch, max_new_tokens=32)
        for i, (p, o) in enumerate(zip(args.batch, outputs)):
            print(f"  [{i}] {p[:40]}... -> {o[:80]}...")

    if args.benchmark:
        import time
        corpus_path = kwargs.pop('corpus', './accepted-001.jsonl')
        print(f"\n  Running compression benchmark (100 docs)...")
        texts = []
        with open(corpus_path) as f:
            for i, line in enumerate(f):
                if i >= 100: break
                texts.append(json.loads(line)['text'])
        t0 = time.time()
        total_o = 0; total_f = 0
        for t in texts:
            o = len(tokenizer.encode(t))
            f = len(wrapper.encode(t))
            total_o += o; total_f += f
        elapsed = time.time() - t0
        comp = (total_o - total_f) / total_o * 100
        spd = (total_o**2) / (total_f**2)
        print(f"  Total orig: {total_o:,} → fused: {total_f:,}")
        print(f"  Compression: {comp:.2f}% | Prefill speedup (O(N²)): {spd:.2f}x")
        print(f"  Time: {elapsed:.2f}s ({elapsed/100*1000:.1f}ms/doc)")
        print(f"  Vocab: base={wrapper.base_vocab_size:,} + fusion={len(wrapper.trie):,}")
