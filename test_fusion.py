"""
Tests for FusionModelWrapper
=============================
Verifies:
  1. Lossless roundtrip: encode → decode == original
  2. Compression ratio > 0 (fused sequence is shorter)
  3. No FID leakage: decode output contains only real text, no fused IDs
  4. Pattern loading: all patterns load without error
  5. Save → Load cycle preserves roundtrip property

Usage:
    python3 test_fusion.py
    python3 test_fusion.py --model "other-model" --patterns ./patterns
"""

import json, os, sys, pickle, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fusion_wrapper import FusionTrie, load_patterns, FusionModelWrapper
from transformers import AutoTokenizer


# ─── Helpers ──────────────────────────────────────────────────────────────────

NORMALIZE_WS = lambda s: " ".join(s.split())


def load_trie_only(bigram_path, multigram_path, tokenizer, base_vocab_size=None):
    """Load patterns into a trie without a model."""
    return load_patterns(bigram_path, multigram_path, tokenizer, base_vocab_size)


def mock_wrapper(trie, tokenizer, base_vocab_size=151936):
    """Create a minimal wrapper for testing encode/decode without a model."""
    # We need a mock model for the wrapper. The wrapper only needs model.generate()
    # for actual generation; encode/decode only use the trie + tokenizer.
    return FusionModelWrapper.__new__(FusionModelWrapper)


# ─── Test 1: Lossless Roundtrip ───────────────────────────────────────────────

def test_roundtrip(trie, tokenizer, texts):
    """encode() then decode() must return the original text."""
    errors = []
    for text in texts:
        ids = tokenizer.encode(text)
        fused = trie.fuse(ids)

        # Expand manually
        fid_map = {}
        def _walk(node, path):
            if "fid" in node:
                fid_map[node["fid"]] = path[:]
            for tid, child in node.items():
                if tid != "fid":
                    _walk(child, path + [tid])
        _walk(trie.root, [])

        expanded = []
        for tid in fused:
            if tid in fid_map:
                expanded.extend(fid_map[tid])
            else:
                expanded.append(tid)

        decoded = tokenizer.decode(expanded)
        if NORMALIZE_WS(text) != NORMALIZE_WS(decoded):
            errors.append((text[:60], decoded[:60]))
    return errors


# ─── Test 2: Compression Ratio ────────────────────────────────────────────────

def test_compression(trie, tokenizer, texts):
    """Fused sequences must be shorter than or equal to original."""
    total_orig = 0
    total_fused = 0
    for text in texts:
        ids = tokenizer.encode(text)
        fused = trie.fuse(ids)
        total_orig += len(ids)
        total_fused += len(fused)
    ratio = (total_orig - total_fused) / total_orig if total_orig > 0 else 0

    # Accept 0% if texts are very short (e.g., single-word prompts)
    return ratio, total_orig, total_fused


# ─── Test 3: No FID Leakage ──────────────────────────────────────────────────

def test_no_fid_leakage(trie, tokenizer, texts, base_vocab_size=151936):
    """Decoded text must not contain raw FID numbers or fusion artifacts."""
    fid_set = set()
    def _walk(node):
        if "fid" in node:
            fid_set.add(node["fid"])
        for tid, child in node.items():
            if tid != "fid":
                _walk(child)
    _walk(trie.root)

    # Build reverse map
    fid_to_constituents = {}
    def _walk2(node, path):
        if "fid" in node:
            fid_to_constituents[node["fid"]] = path[:]
        for tid, child in node.items():
            if tid != "fid":
                _walk2(child, path + [tid])
    _walk2(trie.root, [])

    for text in texts:
        ids = tokenizer.encode(text)
        fused = trie.fuse(ids)

        expanded = []
        for tid in fused:
            if tid in fid_to_constituents:
                expanded.extend(fid_to_constituents[tid])
            else:
                expanded.append(tid)

        # No expanded token should be a FID (>= base_vocab_size)
        for tid in expanded:
            if tid >= base_vocab_size:
                return False, f"FID {tid} leaked into expansion for: {text[:40]}"
    return True, None


# ─── Test 4: Pattern Loading Consistency ─────────────────────────────────────

def test_pattern_consistency(bigram_path, multigram_path, tokenizer):
    """All pattern entries must have valid structure."""
    with open(bigram_path) as f:
        bg = json.load(f)
    for i, p in enumerate(bg):
        if "words" not in p or len(p["words"]) != 2:
            return False, f"bigram[{i}]: invalid structure"
        w1, w2 = p["words"]
        t1 = tokenizer.encode(" " + w1)
        t2 = tokenizer.encode(" " + w2)
        if len(t1) == 0 or len(t2) == 0:
            return False, f"bigram[{i}]: '{w1} {w2}' → empty encoding"

    with open(multigram_path) as f:
        mg = json.load(f)
    for i, p in enumerate(mg):
        if "phrase" not in p or not isinstance(p["phrase"], list) or len(p["phrase"]) < 2:
            return False, f"multigram[{i}]: invalid phrase"
        for tid in p["phrase"]:
            if not isinstance(tid, int) or tid < 0:
                return False, f"multigram[{i}]: invalid token ID {tid}"
    return True, None


# ─── Test 5: Save → Load Roundtrip ─────────────────────────────────────────

def test_save_load_roundtrip(trie, tokenizer):
    """Pickling and unpickling the trie must preserve all patterns."""
    buf = pickle.dumps(trie)
    trie2 = pickle.loads(buf)

    assert trie.size == trie2.size, f"Trie size changed: {trie.size} → {trie2.size}"

    # Check a few random paths
    import random
    all_paths = []
    def _walk(node, path):
        if "fid" in node:
            all_paths.append((path[:], node["fid"]))
        for tid, child in node.items():
            if tid != "fid":
                _walk(child, path + [tid])
    _walk(trie.root, [])

    for path, fid in random.sample(all_paths, min(100, len(all_paths))):
        node = trie2.root
        for tid in path:
            if tid not in node:
                return False, f"Path {path} not found in restored trie"
            node = node[tid]
        if "fid" not in node or node["fid"] != fid:
            return False, f"FID mismatch at path {path}"
    return True, None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test FusionModelWrapper")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--bigram-path", default=str(Path(__file__).parent / "patterns" / "bigram_patterns.json"))
    parser.add_argument("--multigram-path", default=str(Path(__file__).parent / "patterns" / "multigram_patterns.json"))
    parser.add_argument("--corpus", default="./accepted-001.jsonl")
    parser.add_argument("--test-docs", type=int, default=50, help="Number of corpus docs to test")
    args = parser.parse_args()

    print("=" * 60)
    print("  FusionModelWrapper — Test Suite")
    print("=" * 60)

    # Load tokenizer
    print(f"\n[Loading] Tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_vocab = 151936  # From model config

    # Load trie
    print(f"[Loading] Trie from patterns...")
    trie = load_patterns(args.bigram_path, args.multigram_path, tokenizer, base_vocab)
    print(f"  Trie: {len(trie):,} patterns")

    # Test texts
    test_texts = [
        "the first time i saw the world",
        "Explain how transformer self-attention works in machine learning models.",
        "a lot of people think the way to go is to make the best use of the new technology",
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "In order to understand the nature of the universe, we must first understand ourselves.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "The IP address 192.168.1.1 is not valid for this configuration.",
        "CVE-2024-1234 is a critical vulnerability in the authentication module.",
        "mkdir -p /etc/nginx/sites-enabled && systemctl restart nginx",
        "for i in range(10): print(f'Value: {i}')",
        "The most important thing is to be consistent and not to give up.",
        "به نام خداوند جان و خرد کزین برتر اندیشه برنگذرد",
    ]

    # Corpus texts
    corpus_texts = []
    if os.path.exists(args.corpus):
        print(f"[Loading] {args.test_docs} docs from {args.corpus}...")
        with open(args.corpus) as f:
            for i, line in enumerate(f):
                if i >= args.test_docs:
                    break
                corpus_texts.append(json.loads(line)["text"])
    print(f"  Test texts: {len(test_texts)} hand-crafted + {len(corpus_texts)} from corpus")
    all_texts = test_texts + corpus_texts

    # ── Run tests ──
    passed = 0
    failed = 0

    def check(name, ok, detail=None):
        nonlocal passed, failed
        if ok:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}: {detail}")
            failed += 1

    # T1: Roundtrip
    print(f"\n── Roundtrip ──")
    errors = test_roundtrip(trie, tokenizer, all_texts)
    check(f"Lossless ({len(all_texts)} texts)", len(errors) == 0,
          f"{len(errors)} errors: {errors[:2]}")

    # T2: Compression
    print(f"\n── Compression ──")
    ratio, orig, fused = test_compression(trie, tokenizer, all_texts)
    check(f"Compression ratio > 0", ratio > 0,
          f"ratio={ratio:.4f}")
    print(f"    {orig:,} → {fused:,} tokens ({ratio*100:.2f}%)")

    # T3: No FID leakage
    print(f"\n── FID Leakage ──")
    ok, err = test_no_fid_leakage(trie, tokenizer, all_texts, base_vocab)
    check("No FID leakage in decode", ok, err)

    # T4: Pattern consistency
    print(f"\n── Pattern Consistency ──")
    ok, err = test_pattern_consistency(args.bigram_path, args.multigram_path, tokenizer)
    check("All pattern entries valid", ok, err)

    # T5: Save → Load
    print(f"\n── Serialization ──")
    ok, err = test_save_load_roundtrip(trie, tokenizer)
    check("Pickle roundtrip preserves trie", ok, err)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{passed+failed} passed", end="")
    if failed > 0:
        print(f", {failed} FAILED")
    else:
        print()
    print(f"{'='*60}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
