# DOCS — FusionModelWrapper

## Theory

### O(N²) Problem

Self-attention computes:

```
Attention(Q,K,V) = softmax(QK^T / √d) · V
```

The QK^T matrix is `[N × N]`, cost scales as **O(N²)** where N = sequence length.

**Crucially, this applies only during prefill (encoding the entire input at once).**
Autoregressive decoding processes one token at a time (O(N) per step) and is
not accelerated by token fusion.

```
If N = 1000 → prefill cost = 1,000,000 units
If N = 855  → prefill cost =   731,025 units (1.17x)
```

### Why Token Fusion Works

GPT-4/Qwen BPE tokenizers encode text at subword level. ~40% of tokens are
function words surrounding content words:

| Pattern | Tokens | After Fusion |
|---------|--------|-------------|
| `of the` | 2 | 1 |
| `in order to` | 3 | 1 |
| `i saw` | 2 | 1 |

### What Fusion Does NOT Do

- **No decoding speedup** — each generated token still goes through one forward pass
- **No quality improvement** — random init means logits are uncalibrated
- **No fine-tuning needed for roundtrip** — encode/decode is lossless by construction

## Algorithm: Greedy Longest-Match

FusionTrie implements greedy longest-match:

1. Walk tokens left-to-right
2. At each position, descend into Trie as far as possible
3. Fuse the longest matching span into one token ID
4. Continue after the fused span
5. If no match, keep original token, advance by 1

O(M) time, no backtracking.

```
Input:  [A, B, C, D, E, F]
Trie:   [A,B]→id_1, [B,C]→id_2, [D,E,F]→id_3

Step 1: A→B found → fuse → [id_1, ...]
Step 2: C→no match → keep [C, ...]
Step 3: D→E→F found → fuse → [id_3, ...]
Output: [id_1, C, id_3]
```

## Pattern Mining Pipeline

### Phase 1: Bigram Collection (tokenizer-agnostic)
- 10K docs from a large text corpus (e.g., Dolma, C4, or your own data)
- Text-level word pairs: `stopword + content_word`
- Top 20K by frequency

### Phase 2: Multi-gram Collection (model-specific)
- 3K docs, using Qwen2.5 tokenizer
- Token-level n-grams (3-6 tokens) with frequency ≥ 3
- Scored by `(n-1) × freq` (token savings)
- Safety filter removes: digits, paths, shell chars
- Top ~13K patterns retained

### Phase 3: Trie Construction
- Bigrams: stored as text, encoded with base tokenizer at load time → works with any model
- Multigrams: stored as Qwen token IDs → exact match only with Qwen
- Total: 33,303 patterns

## Embedding Initialization

**Random init from existing distribution** (not mean):

```python
existing = embeddings[:base_vocab_size]
mean = existing.mean(dim=0)
std = existing.std(dim=0)
embeddings[new_tokens] = mean + noise * std
```

> **Why not mean init?** `mean("New", "York")` produces a vector in the
> semantic middle, NOT a vector representing "New York". The model has never
> seen this interpolated vector. The softmax logits are entirely uncalibrated.
> Random init is equally uncalibrated but avoids false confidence in
> semantically "correct" vectors.

**LoRA fine-tuning is REQUIRED** for usable generation quality.

## API Reference

### `from_pretrained(model, ...)`

| Param | Default | Description |
|-------|---------|-------------|
| `model_name_or_path` | required | HuggingFace model name or path |
| `bigram_path` | `./bigram_patterns.json` | Bigram pattern file |
| `multigram_path` | `./multigram_patterns.json` | Multi-gram pattern file |
| `device_map` | `"auto"` | Device placement |
| `torch_dtype` | `torch.float16` | Model precision |

Returns: `(FusionModelWrapper, tokenizer)`

### `FusionModelWrapper`

| Method | Description |
|--------|-------------|
| `encode(text)` | Fuse text → shorter token IDs (prefill speedup) |
| `decode(ids)` | Expand fused IDs → text |
| `generate(prompt, **kwargs)` | Fuse → generate → decode |
| `generate_batch(prompts, ...)` | Batch generation with padding |
| `save_pretrained(path)` | Save full checkpoint |
| `to(device)` | Move model to device |

### `from_fused_pretrained(path, ...)`

Load a previously saved fused checkpoint.

## Performance

### Compression by Document Type

| Type | Original | Fused | Compression | Prefill Speedup |
|------|:-------:|:-----:|:----------:|:--------------:|
| Stopword-heavy (news) | 1,000 | ~830 | 17% | 1.45x |
| Technical (code docs) | 1,000 | ~890 | 11% | 1.26x |
| Mixed (average) | 1,000 | ~855 | 14.5% | 1.37x |

### Compression vs Pattern Count

| Patterns | Compression | Prefill Speedup | VRAM (d=1536, FP16) |
|:--------:|:----------:|:--------------:|:-----------------:|
| 10K | 8% | 1.18x | 0.031 GB |
| 20K | 11% | 1.26x | 0.062 GB |
| 33K | 14.5% | 1.37x | 0.102 GB |
| 50K | 17% | 1.45x | 0.154 GB |

### VRAM Cost

| Model | d_model | Extra VRAM (33K tokens, FP16) |
|-------|:------:|:---------------------------:|
| 1.5B (Qwen) | 1536 | **0.102 GB** |
| 7B (Llama) | 4096 | **0.271 GB** |
| 13B (Llama) | 5120 | **0.339 GB** |
| 70B (Llama) | 8192 | **0.542 GB** |

## Pattern Storage Format

### `patterns/bigram_patterns.json`

Actually stored at root level in the repo: `bigram_patterns.json`.

```json
[
  {"words": ["the", "world"], "frequency": 16228, "id": 0},
  {"words": ["to", "make"], "frequency": 19785, "id": 1},
  ...
]
```

### `patterns/multigram_patterns.json`

Also at root level: `multigram_patterns.json`.

```json
[
  {"phrase": [264, 2763, 315], "frequency": 47, "savings": 94, "id": 0},
  {"phrase": [11, 323, 279], "frequency": 38, "savings": 76, "id": 1},
  ...
]
```

## File Structure

```
Enhance-models/
├── fusion_wrapper.py         ← Main library (production-ready)
├── mine_patterns.py          ← Pattern miner script
├── lora_finetune.py          ← LoRA fine-tuning script
├── test_fusion.py            ← Test suite (5 tests)
├── cross_model_bench.py      ← Cross-model benchmark (6 families)
├── bigram_patterns.json      ← 20K bigram patterns (text-level)
├── multigram_patterns.json   ← 13K multi-gram patterns (Qwen tokens)
├── pyproject.toml            ← pip install config
├── README.md                 ← Quick start guide
├── DOCS.md                   ← This file
├── LICENSE                   ← MIT
└── .gitignore
```

## Limitations

1. **Prefill-only speedup**: Autoregressive decoding dominates wall-clock time
   on GPU. True end-to-end speedup requires fusion of KV-cache or speculative
   decoding.

2. **Random init = uncalibrated**: Without LoRA fine-tuning, the first
   generation after a fused token may produce unexpected continuations.

3. **Model-specific multigrams**: The 13K multi-gram patterns are mined with
   Qwen2.5 tokenizer and won't match other models unless re-mined.

4. **Bigram first-token only**: `encode(" " + w1)` may return multiple subword
   tokens for rare words, but only `t1[0]` is used as the match key. This is
   conservative — roundtrip remains lossless — but means multi-token content
   words like "Württemberg" → `[t1, t2]` will not fuse via bigram.
   Stopwords (the, of, in) are reliably single-token in BPE tokenizers,
   so ~95% of bigram patterns match correctly.
