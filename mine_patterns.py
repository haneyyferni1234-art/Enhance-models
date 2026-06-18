"""
Pattern Miner for FusionModelWrapper.
======================================
Mines bigram + multi-gram patterns from a corpus using the *target model's*
own tokenizer (not tiktoken), so token IDs match exactly at runtime.

Usage:
    # Default: 10K docs for bigrams, 5K for multigrams
    python3 mine_patterns.py

    # Full-scale
    python3 mine_patterns.py --bigram-docs 50000 --multigram-docs 10000
"""

import json, re, math
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer


# ── Stopword list (English, function-word-focused) ──────────────────────────
STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "for", "on", "with", "at", "by",
    "and", "or", "not", "no", "but", "if", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "shall", "should", "may", "might",
    "must", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "some", "any", "each", "every",
    "all", "both", "few", "many", "much", "more", "most", "other",
    "such", "as", "than", "so", "very", "too", "quite", "just",
    "about", "around", "above", "below", "between", "through", "during",
    "before", "after", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "which", "who", "whom", "what",
    "into", "onto", "upon", "from", "within", "without",
    "because", "since", "while", "though", "although", "until",
    "like", "also", "only", "still", "even", "well", "back",
}


def text_to_words(text):
    """Split text into lowercase words (simple whitespace + punctuation)."""
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def is_safe(phrase_tokens):
    """Safety filter: reject patterns containing digits, paths, shell chars."""
    DANGER = re.compile(r'[<>|&;$`(){}[\]\\/#]|\d')
    for t in phrase_tokens:
        if isinstance(t, str) and DANGER.search(t):
            return False
    return True


# ── Bigram Mining (text-level, tokenizer-agnostic) ──────────────────────────

def mine_bigrams(texts, top_k=20000, min_freq=5):
    """
    Mine word-level bigrams: (stopword + content_word) pairs.
    Stored as text so they can be re-encoded with any tokenizer at load time.
    """
    bigram_counter = Counter()

    for text in texts:
        words = text_to_words(text)
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            if w1 in STOPWORDS and w2 not in STOPWORDS:
                bigram_counter[(w1, w2)] += 1

    top = bigram_counter.most_common(top_k)
    patterns = [{"words": [w1, w2], "frequency": freq, "id": i}
                for i, ((w1, w2), freq) in enumerate(top)]
    print(f"  Bigrams mined: {len(patterns):,} patterns (from {len(bigram_counter):,} unique)")
    return patterns


# ── Multi-gram Mining (token-level, model-specific) ─────────────────────────

def tokenize_safe(tokenizer, text):
    """Tokenize and return IDs, filtering out special token IDs."""
    ids = tokenizer.encode(text)
    spec = set(tokenizer.all_special_ids or [])
    return [i for i in ids if i not in spec]


def mine_multigrams(texts, tokenizer, top_k=15000, min_freq=3, max_ngram=6):
    """
    Mine token-level n-grams (3-6 grams) using the model's tokenizer.
    Stored as token-ID lists for exact matching.
    """
    counter = Counter()
    total_docs = 0

    for text in texts:
        ids = tokenize_safe(tokenizer, text)
        if len(ids) < 3:
            continue
        total_docs += 1

        for n in range(3, min(max_ngram, len(ids)) + 1):
            seen_this_doc = set()
            for i in range(len(ids) - n + 1):
                gram = tuple(ids[i:i+n])
                if gram in seen_this_doc:
                    continue
                seen_this_doc.add(gram)
                counter[gram] += 1

    # Filter by frequency
    candidates = [(gram, freq) for gram, freq in counter.items() if freq >= min_freq]
    print(f"  Multi-grams mined: {len(candidates):,} pass min_freq={min_freq}")

    # Score by token savings: (n-1 tokens saved per match) * freq
    scored = [(gram, freq, (len(gram) - 1) * freq) for gram, freq in candidates]
    scored.sort(key=lambda x: -x[2])

    top = scored[:top_k]
    patterns = [{"phrase": list(gram), "frequency": freq, "savings": sav, "id": i}
                for i, (gram, freq, sav) in enumerate(top)]

    total_savings = sum(sav for _, _, sav in top)
    print(f"  Multi-grams selected: {len(top):,} patterns")
    print(f"  Estimated token savings per doc: {total_savings / max(len(texts), 1):.1f}")
    return patterns


# ── Safety Filter (remove patterns with digits, paths, code) ───────────────

def safety_filter_multigrams(patterns, tokenizer):
    """Remove patterns whose decoded text contains unsafe characters."""
    filtered = []
    removed = 0
    for p in patterns:
        decoded = tokenizer.decode(p["phrase"])
        if is_safe([decoded]):
            filtered.append(p)
        else:
            removed += 1
    print(f"  Safety filter removed {removed} unsafe patterns")
    return filtered


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mine fusion patterns from corpus")
    parser.add_argument("--corpus", default="./accepted-001.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--bigram-docs", type=int, default=10000, help="Docs to sample for bigrams")
    parser.add_argument("--multigram-docs", type=int, default=5000, help="Docs to sample for multigrams")
    parser.add_argument("--top-bigrams", type=int, default=20000)
    parser.add_argument("--top-multigrams", type=int, default=15000)
    parser.add_argument("--output-dir", default="./patterns")
    args = parser.parse_args()

    print("=" * 60)
    print("  Fusion Pattern Miner")
    print("=" * 60)

    # Load tokenizer
    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"     Vocab size: {tokenizer.vocab_size:,}")

    # Read corpus
    print(f"\n[2] Reading corpus: {args.corpus}")
    all_texts = []
    with open(args.corpus) as f:
        for line in f:
            all_texts.append(json.loads(line)["text"])
    print(f"     Total docs: {len(all_texts):,}")

    bigram_texts = all_texts[:args.bigram_docs]
    multigram_texts = all_texts[:args.multigram_docs]

    # Mine bigrams
    print(f"\n[3] Mining bigrams from {len(bigram_texts):,} docs...")
    bigrams = mine_bigrams(bigram_texts, top_k=args.top_bigrams)

    # Mine multigrams
    print(f"\n[4] Mining multigrams from {len(multigram_texts):,} docs...")
    multigrams = mine_multigrams(multigram_texts, tokenizer, top_k=args.top_multigrams)

    # Safety filter
    print(f"\n[5] Safety filtering multigrams...")
    multigrams = safety_filter_multigrams(multigrams, tokenizer)

    # Save
    print(f"\n[6] Saving patterns...")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bg_path = out / "bigram_patterns.json"
    with open(bg_path, "w") as f:
        json.dump(bigrams, f, ensure_ascii=False)
    print(f"     Bigrams: {len(bigrams):,} → {bg_path}")

    mg_path = out / "multigram_patterns.json"
    with open(mg_path, "w") as f:
        json.dump(multigrams, f, ensure_ascii=False)
    print(f"     Multigrams: {len(multigrams):,} → {mg_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Done. Bigrams: {len(bigrams):,}, Multigrams: {len(multigrams):,}")
    print(f"  Total patterns: {len(bigrams) + len(multigrams):,}")
    print(f"  Output: {out.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
