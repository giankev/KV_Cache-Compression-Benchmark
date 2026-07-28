# KV Cache Compression Benchmark

Small Hugging Face benchmark for post-prefill KV-cache compression with
Qwen2.5. It compares:

- no compression;
- low-L2, random, and high-L2 token selection;
- KeyDiff key-similarity token selection;
- SnapKV attention-based token selection.

The L2, KeyDiff, and SnapKV implementations are intentionally separate. The
online language-modelling benchmark remains available in
`scripts/run_online_lm.py`.

## Setup

Python 3.10-3.14 is supported.

```bash
python -m pip install -e ".[test]"
```

For Kaggle, use the pinned environment:

```bash
python -m pip install -r requirements-kaggle.txt
```

## Professor-style passkey benchmark

The benchmark follows the single-passkey task in
[`eval_passkey.py`](https://github.com/alessiodevoto/l2compress/blob/main/eval_passkey.py)
and its greedy cache generation flow in
[`gen_utils.py`](https://github.com/alessiodevoto/l2compress/blob/main/gen_utils.py):

- one integer passkey between 1 and 50000;
- repeated irrelevant text around the information line;
- one random information position determined by each seed;
- the passkey written twice in the information line;
- the question at the end of the prompt;
- exact token match over the answer length;
- exact-match accuracy aggregated over the seeds that actually run.

The prompt is assembled directly from separately tokenized component IDs. It
has exactly `context_length` tokens and is never decoded and re-tokenized.
`random.Random(seed)` isolates prompt generation from global random state. The
terminal question is:

```text
Question: What is the pass key?
Answer with only the number.
The pass key is ␠
```

Here `␠` makes the trailing space visible; the actual prompt ends with
`The pass key is `. Nothing follows it before generation, and the expected
number is tokenized separately for exact matching.

### Baseline calibration

Start with the easier 4k sanity check:

```bash
python scripts/run_l2_passkey.py \
  --context-lengths 4096 \
  --seeds 0 \
  --output-prefix l2_passkey_sanity_4k
```

Then repeat at the final default context length:

```bash
python scripts/run_l2_passkey.py \
  --context-lengths 8192 \
  --seeds 0 \
  --output-prefix l2_passkey_sanity_8k
```

Inspect the `no_compression` line and raw row at both lengths. A failed baseline
is retained, and the compressed configurations are skipped for that seed. Do
not interpret compression results until both baselines have been verified.

### Complete 8k experiment

```bash
python scripts/run_l2_passkey.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --context-lengths 8192 \
  --seeds 0 1 2 \
  --keep-ratio 0.10 \
  --skip-layers 0 1 \
  --chunk-size 512 \
  --output-prefix l2_passkey_3b_8k_keep10

python scripts/run_keydiff_passkey.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --context-lengths 8192 \
  --seeds 0 1 2 \
  --keep-ratio 0.10 \
  --skip-layers 0 1 \
  --chunk-size 512 \
  --output-prefix keydiff_passkey_3b_8k_keep10

python scripts/run_snapkv_passkey.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --context-lengths 8192 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --target-cache-tokens 1024 \
  --observation-window-size 32 \
  --pooling-kernel-size 5 \
  --pooling-mode max \
  --chunk-size 512 \
  --output-prefix snapkv_passkey_3b_8k
```

The default L2 experiment evaluates `no_compression`, `low_l2`, `random`, and
`high_l2` on seeds 0, 1, and 2. Every compressed configuration uses the
requested `--keep-ratio` (10% by default) for each non-skipped cache layer:

```text
3 seeds x 4 configurations = at most 12 runs
```

The effective number can be lower because a failed baseline skips the three
compressed configurations for that seed. The SnapKV runner evaluates only
`no_compression` and `snapkv`, with a 32-token observation window and a
1024-token target cache by default.

All three passkey runners compress every layer by default. Use
`--skip-layers` to explicitly list layers that must remain uncompressed; for
example, `--skip-layers 0 1` reproduces the previous L2 comparison protocol.
The selected runtime value is recorded in metadata and raw results.

The KeyDiff runner evaluates `no_compression` and `keydiff` with the same
ratio and seeds as L2. KeyDiff does not intrinsically require skipped layers.
It retains keys with low cosine similarity to an anchor obtained by averaging
normalized cached keys. This implements only the KeyDiff scoring and eviction
criterion, not the paper's full block-wise inference or BlockPress protocol.

Each prompt is built once per context length and seed. The uncompressed
baseline runs first. If it fails, that baseline row is saved and compressed
configurations are not executed for the case.

For later experiments, change only the context length and output prefix:

```bash
python scripts/run_l2_passkey.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --context-lengths 16384 \
  --seeds 0 1 2 \
  --keep-ratio 0.10 \
  --skip-layers 0 1 \
  --chunk-size 512 \
  --output-prefix l2_passkey_3b_16k_keep10

python scripts/run_snapkv_passkey.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --context-lengths 16384 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --target-cache-tokens 1024 \
  --observation-window-size 32 \
  --pooling-kernel-size 5 \
  --pooling-mode max \
  --chunk-size 512 \
  --output-prefix snapkv_passkey_3b_16k
```

Use `--context-lengths 32768` for the corresponding 32k experiment.

### Outputs

Every completed run immediately checkpoints:

```text
results/<output-prefix>_raw.csv
```

The raw columns are:

```text
model_name, method, config, context_length, keep_ratio,
target_cache_tokens, observation_window_size, pooling_kernel_size,
pooling_mode, skip_layers, seed, actual_depth, target, prediction, correct,
cache_before_mb, cache_after_mb, memory_saved_mb, memory_saved_percent,
elapsed_seconds
```

The summary contains:

```text
model_name, method, config, context_length, keep_ratio,
target_cache_tokens, num_examples, accuracy,
mean_cache_before_mb, mean_cache_after_mb, mean_memory_saved_mb,
mean_memory_saved_percent, mean_elapsed_seconds
```

Cache memory is measured from the real K/V tensors immediately after prefill
and immediately after optional compression, before answer generation.
`memory_saved_mb` is their difference. `memory_saved_percent` is derived from
those two full-cache measurements rather than from the nominal keep ratio.

For L2 and KeyDiff, `keep_ratio` is the requested ratio (`1.0` for the
baseline) and `target_cache_tokens` is the resulting capacity of each
compressed layer. Pooling fields are empty because these methods do not use
pooling. For SnapKV, `keep_ratio` is the effective ratio
`target_cache_tokens / context_length`, and the target capacity, observation
window, pooling kernel, and pooling mode are recorded explicitly.
`skip_layers` always records the actual CLI selection and is empty by default
for every passkey runner. When methods compress different layer sets, compare
their real `memory_saved_percent`, not only their keep ratios.

Torch and Transformers versions, dtype, device map, skip layers, seeds, and
method parameters are stored once in the metadata JSON. SnapKV metadata also
records `effective_keep_ratio` by context length; its `target_cache_tokens`
includes the observation window (for example, 992 selected prefix tokens plus
32 observation tokens equals a 1024-token target).

`elapsed_seconds` is end-to-end evaluation time: prefill, optional compression,
and answer generation. It is not decode-only time or a throughput metric. The
console summary shows only the compact accuracy, memory, ratio, and elapsed-time
columns; full experimental parameters remain in the raw CSV and metadata.

### Plot retrieval accuracy

```bash
python scripts/plot_retrieval.py \
  --input-csv results/l2_passkey_3b_8k_keep10_raw.csv \
  --output results/l2_passkey_3b_8k_keep10_accuracy.png \
  --title "L2 passkey retrieval accuracy"
```

The plot groups raw results by configuration and context length. It uses a
headless Matplotlib backend and is suitable for Kaggle notebooks.

## Smoke tests

These scripts load Qwen models and are therefore separate from the unit suite:

```bash
python scripts/smoke_test_qwen.py
python scripts/smoke_test_snapkv_qwen.py
```

The first checks logical-position decoding with heterogeneous layer lengths.
The second checks Qwen GQA attention aggregation, SnapKV compression, and
post-compression decoding. SnapKV uses eager attention and retains support for
models sharded by `device_map="auto"`.

## Online language modelling

Online LM is a reduced reproduction inspired by the Wikipedia language
modelling experiment in *A Simple and Effective L2 Norm-Based Strategy for KV
Cache Compression*. It compares `low_l2`, `keydiff`, `random`, `high_l2`, and
`snapkv` against `no_compression` on:

- one deterministic 8,192-token stream from the WikiText-2 test split;
- a fixed maximum KV-cache capacity of 2,000 tokens;
- 32-token causal blocks, with loss and next-token accuracy computed at every
  position;
- cumulative checkpoints every 512 processed tokens.

Run the benchmark and plot its main cumulative log-PPL curve with:

```bash
python scripts/run_online_lm.py
python scripts/plot_online_lm.py
```

By default every compression method operates on every model layer. To preserve
layers 0 and 1 for all compressed methods in the same controlled run, pass the
shared experimental parameter explicitly:

```bash
python scripts/run_online_lm.py --skip-layers 0 1
```

Layer skipping is therefore a benchmark control, not an implicit requirement
of KeyDiff, SnapKV, or any other eviction strategy.

The benchmark writes `results/online_lm_curve.csv`,
`results/online_lm_summary.csv`, and `results/online_lm_metadata.json`. The plot
defaults to `results/online_lm_log_ppl.png`, includes the KeyDiff curve
automatically, and marks the 2,000-token KV budget. An optional accuracy plot
can be produced with:

```bash
python scripts/plot_online_lm.py \
  --accuracy-output results/online_lm_accuracy.png
```

For online SnapKV, each incoming 32-token block acts as the observation window
when compression is required. This is a simple chunked adaptation for the
shared online benchmark; it is not presented as the original SnapKV paper
protocol.

## Main files

- `src/l2kv/passkey.py`: exact professor-style prompt construction.
- `src/l2kv/retrieval_eval.py`: shared prefill, compression evaluation, exact
  answer generation, cache measurement, checkpointing, and summaries.
- `scripts/run_l2_passkey.py`: L2-only runner.
- `scripts/run_keydiff_passkey.py`: baseline and KeyDiff runner.
- `scripts/run_snapkv_passkey.py`: baseline and SnapKV runner.
- `scripts/plot_retrieval.py`: generic accuracy-versus-context plot.
- `scripts/plot_online_lm.py`: cumulative online LM log-PPL plot.
- `src/l2kv/l2_compression.py`: low-L2, high-L2, and random cache policies.
- `src/l2kv/keydiff_compression.py`: KeyDiff key-similarity cache policy.
- `src/l2kv/snapkv_compression.py`: SnapKV scoring and cache rewrite.

Implementation details are in
[`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md) and
[`docs/SNAPKV_NOTES.md`](docs/SNAPKV_NOTES.md).

## Scope

In the passkey benchmark, compression happens after the full prompt prefill and
does not reduce prefill compute or peak prompt memory. The online LM benchmark
instead applies its fixed budget after each 32-token block once the cache would
exceed 2,000 tokens. Results are educational reduced benchmarks, not
comprehensive reproductions of either reference repository.
