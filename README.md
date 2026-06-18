# FusionModelWrapper — Lossless Token Fusion (Prefill-Only)

تسريع مرحلة ترميز الإدخال (Prefill) في نماذج LLM بنسبة **تصل إلى 1.46x** عبر دمج الرموز اللغوية المكررة (Lossless) بدون المساس بدقة النموذج.

## المبدأ

~40% من الرموز في النصوص هي كلمات وظيفية مكررة. FusionModelWrapper يدمجها مع الكلمات المجاورة في رموز مركبة واحدة، مما يقلل طول تسلسل الإدخال وبالتالي يسرع self-attention خلال مرحلة prefill فقط.

```
Input:  "the" + " first" + " time" + " i" + " saw" + " the" + " world"  = 7 tokens
Fused:  [the_first]  +  [time_i_saw]  +  [the_world]                    = 3 tokens
```

> ⚠ **هام**: التسريع ينطبق فقط على **مرحلة prefill** (ترميز الإدخال). التوليد التتابعي (autoregressive decoding) غير متأثر.

> ⚠ **هام**: التهيئة **عشوائية** (Random Init) - الـ model يحتاج **LoRA fine-tuning** لنتائج إنتاجية.

## النتائج

| المقياس | القيمة |
|---------|--------|
| ضغط الترميز | 11-17% |
| تسريع prefill (O(N²)) | 1.46-1.46x |
| خسارة البيانات | 0% (Lossless) |
| أنماط الدمج | 33,303 (20K bigram + 13.3K multigram) |
| VRAM إضافي | 0.108 GB (d=1536, FP16) |

## المتطلبات

```bash
pip install torch transformers
```

## الاستخدام

### 1. تحميل النموذج والدمج

```python
from fusion_wrapper import from_pretrained

wrapper, tokenizer = from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
```

### 2. توليد نص

```python
output = wrapper.generate("Explain transformer attention", max_new_tokens=256)
print(output)

# Batch
outputs = wrapper.generate_batch([
    "What is self-attention?",
    "Explain RAG in AI",
    "How do transformers work?"
], max_new_tokens=128)
```

### 3. حفظ وتحميل checkpoint

```python
wrapper.save_pretrained("./fused-model")

from fusion_wrapper import from_fused_pretrained
wrapper2, tok2 = from_fused_pretrained("./fused-model")
```

### CLI

```bash
# إنشاء checkpoint
python3 fusion_wrapper.py --model Qwen/Qwen2.5-1.5B-Instruct --save ./fused-qwen

# تحميل وتوليد
python3 fusion_wrapper.py --load ./fused-qwen --prompt "Explain self-attention"

# توليد batch
python3 fusion_wrapper.py --load ./fused-qwen --batch "What is AI?" "Explain RAG"

# Benchmark
python3 fusion_wrapper.py --load ./fused-qwen --benchmark
```

## كيف يعمل؟

```
Input → Tokenizer → [token IDs] → FusionTrie (greedy longest-match)
                                    ↓
                             [fused IDs] → model.generate() (prefill أسرع)
                                    ↓
                             FusionTrie.expand → decode → Output
```

### Flistructure

```
AI-MODEL/
├── fusion_wrapper.py         ← المكتبة الرئيسية
├── mine_patterns.py          ← استخراج الأنماط من الكوربس
├── patterns/
│   ├── bigram_patterns.json  ← 20K bigram (نصي, لأي tokenizer)
│   └── multigram_patterns.json ← 13K multigram (خاص بـ Qwen tokens)
├── README.md                 ← هذا الملف
├── DOCS.md                   ← توثيق كامل
├── accepted-001.jsonl        ← الكوربس (562K وثيقة)
└── accepted-002.txt          ← الكوربس بصيغة نصية
```

### ملاحظات مهمة

- **Mean Initialization** أزيلت: `mean("New", "York") ≠ "New York"`. المتجهات الجديدة تبدأ عشوائياً من توزيع الـ embeddings الأصلية.
- **LoRA ضروري** لضبط logits الرموز الجديدة. شاهد `lora_finetune.py`.
- **الأنماط** تم استخراجها باستخدام tokenizer النموذج نفسه (ليس tiktoken)، مما يضمن تطابق المعرفات.
- **الأمان**: جميع الأنماط مرشحة لإزالة الأرقام، المسارات، ورموز shell.
