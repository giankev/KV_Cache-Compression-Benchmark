# KV Cache Compression Benchmark

Small, script-first benchmark for L2-norm and SnapKV KV-cache compression with
Hugging Face `DynamicCache`. The main model is `Qwen/Qwen2.5-3B-Instruct`;
layers 0 and 1 remain uncompressed.

The implementation follows the core observation of
[`alessiodevoto/l2compress`](https://github.com/alessiodevoto/l2compress):
`low_l2` retains keys with the smallest L2 norm. The same temporal indices are
applied to values and restored to chronological order before the cache is used
again.

## Kaggle Quickstart

Enable Internet in the Kaggle notebook settings and select a GPU accelerator.
Then run these cells:

```python
!git clone https://github.com/giankev/KV_Cache-Compression-Benchmark.git
%cd KV_Cache-Compression-Benchmark
!pip install -r requirements-kaggle.txt
!pip install -e .
```

Run the 0.5B smoke test before the 3B benchmark:

```python
!python scripts/smoke_test_qwen.py
!python scripts/run_basic_passkey.py
!python scripts/run_online_lm.py
```

The first execution downloads the Qwen model and, for online LM evaluation,
WikiText. No notebook is required: every experiment is an ordinary Python
script.

Existing visualizations can also be generated from Kaggle cells:

```python
!python scripts/show_attention_l2.py
!python scripts/show_alr_heatmap.py
```

CSV and JSON outputs are written under `results/`; visualization scripts write
PNG files there as well. Each executable records the current software/model
settings in `results/run_metadata.json` (the latest run replaces that file).

## Passkey benchmark

The default demonstration uses:

- model: `Qwen/Qwen2.5-3B-Instruct`;
- context lengths: 8192 and 32768 tokens;
- needle depths: 0.25, 0.50, and 0.75;
- seed: 0;
- configurations: `no_compression`, `low_l2_keep50`, `low_l2_keep10`,
  `random_keep50`, and `high_l2_keep50`.

This gives `2 * 3 * 1 * 5 = 30` runs. The prompt is assembled directly from
token IDs, so `context_len_actual` equals its target exactly. Compression is
performed once after context prefill. Memory before and immediately after that
compression is measured from the actual cache tensors; final generation memory
is stored separately.

With one seed and only three depths this is a demonstrative university
benchmark, not a statistically conclusive evaluation.

For a quick reduced check of all five strategies:

```bash
python scripts/run_basic_passkey.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --context-lengths 256 \
  --depths 0.5 \
  --seeds 0 \
  --prune-after 0 \
  --output-prefix basic_passkey_reduced
```

The default passkey outputs are:

- `results/basic_passkey_raw.csv`;
- `results/basic_passkey_summary.csv`.

## Generate the passkey heatmap

Generate an accuracy-by-depth heatmap from the raw passkey CSV, which contains
the `depth_target` column:

```python
!python scripts/plot_passkey_heatmap.py \
  --input-csv results/passkey_3b_32k_depths_raw.csv \
  --output results/passkey_3b_32k_depths_heatmap.png \
  --title "Passkey retrieval accuracy by depth — Qwen2.5-3B, 32k context"
```

The script prints the aggregated accuracy table before saving the PNG. If more
than one seed is present, each cell contains the mean accuracy for that
configuration and depth.

## SnapKV key-value retrieval benchmark

SnapKV uses attention from a final observation window to select important
prefix positions independently for each native KV head. The implementation is
compatible with Qwen2.5 Grouped-Query Attention: votes from query heads that
share a KV head are aggregated without permanently replicating the cache. See
`docs/SNAPKV_NOTES.md` for the algorithm, fair-capacity convention, and
limitations.

Run the unit tests and the focused 0.5B SnapKV smoke test on Kaggle:

```python
!python -m pytest -q
!python scripts/smoke_test_snapkv_qwen.py
```

Run the reduced benchmark across all seven configurations:

```python
!python scripts/run_kv_retrieval.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --context-lengths 256 \
  --depths 0.5 \
  --seeds 0 \
  --observation-window-size 16 \
  --pooling-kernel-size 5 \
  --output-prefix kv_retrieval_reduced
```

Run the 3B, 32k benchmark at three target depths:

```python
!python scripts/run_kv_retrieval.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --context-lengths 32000 \
  --depths 0.25 0.50 0.75 \
  --seeds 0 \
  --observation-window-size 64 \
  --pooling-kernel-size 5 \
  --output-prefix kv_retrieval_3b_32k
```

The raw output includes `config`, `depth_target`, and `correct`, so the same
heatmap script used for the passkey benchmark can visualize it:

```python
!python scripts/plot_passkey_heatmap.py \
  --input-csv results/kv_retrieval_3b_32k_raw.csv \
  --output results/kv_retrieval_3b_32k_heatmap.png \
  --title "Key-value retrieval accuracy by depth - Qwen2.5-3B, 32k context"
```

The 3B command is intended for Kaggle or another suitable GPU environment, not
for a local CPU run.

## Online language modeling and ALR

`scripts/run_online_lm.py` evaluates WikiText token by token with a fixed cache
budget and writes `results/online_lm_summary.csv`. Its average memory comparison
uses a theoretical uncompressed baseline at the same logical step, rather than
the final baseline for every step.

`scripts/run_alr_scan.py` remains an exploratory tool. It measures attention/L2
only over tokens that were present before the decode query and writes its CSV
files under `results/`. It does not alter the fixed `(0, 1)` skip-layer choice
used by the benchmark.

## Reproducibility and cache positions

The selected dependency set is pinned in `requirements-kaggle.txt`, including
`transformers==4.57.6`, whose `DynamicCache.layers` API is used directly.
Random passkey numbers and random cache selection use local seed-derived random
sources; the compression functions never call global `torch.manual_seed`.

After pruning, the logical token position is kept separate from every layer's
physical cache length. With batch size 1 and no padding, question/answer tokens
are forwarded one at a time with explicit `position_ids` and `cache_position`.
No all-ones attention mask is inferred from the longer layer 0. This supported
path is checked by `scripts/smoke_test_qwen.py`.

## Tests

```bash
python -m pytest -q
python scripts/smoke_test_qwen.py
python scripts/smoke_test_snapkv_qwen.py
```

Unit tests use a synthetic DynamicCache-compatible object and do not download a
model. The smoke test downloads `Qwen/Qwen2.5-0.5B-Instruct`.

## Structure

- `src/l2kv/cache_compression.py` - validated in-place cache compression.
- `src/l2kv/passkey.py` - exact-token prompt construction and greedy decoding.
- `src/l2kv/snapkv.py` - observation-window voting and GQA-aware compression.
- `src/l2kv/kv_retrieval.py` - exact-token synthetic key-value prompts.
- `src/l2kv/position_utils.py` - shared logical-position helpers.
- `src/l2kv/cache_metrics.py` - actual and theoretical cache sizes.
- `src/l2kv/alr.py` - exploratory ALR calculation.
- `scripts/run_basic_passkey.py` - main passkey benchmark.
- `scripts/run_kv_retrieval.py` - SnapKV and baseline retrieval benchmark.
- `scripts/run_online_lm.py` - online WikiText benchmark.
- `scripts/smoke_test_qwen.py` - heterogeneous-layer forward smoke test.
- `scripts/smoke_test_snapkv_qwen.py` - GQA SnapKV integration smoke test.
- `docs/IMPLEMENTATION_NOTES.md` - concise implementation/exam notes.
- `docs/SNAPKV_NOTES.md` - paper summary, adaptations, and limitations.
