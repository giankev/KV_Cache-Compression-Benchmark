# KV Cache Compression Benchmark

Small Hugging Face benchmark for post-prefill KV-cache compression with
Qwen2.5. It compares:

- no compression;
- low-L2, random, and high-L2 token selection;
- SnapKV attention-based token selection.

The L2 and SnapKV implementations are intentionally separate. The online
language-modelling benchmark remains available in `scripts/run_online_lm.py`.

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
`random.Random(seed)` isolates prompt generation from global random state.

### Quick checks

L2 with one seed:

```bash
python scripts/run_l2_passkey.py \
  --seeds 0 \
  --output-prefix l2_passkey_sanity
```

SnapKV with one seed:

```bash
python scripts/run_snapkv_passkey.py \
  --seeds 0 \
  --output-prefix snapkv_passkey_sanity
```

### Complete 8k experiment

```bash
python scripts/run_l2_passkey.py \
  --output-prefix l2_passkey_3b_8k_keep10

python scripts/run_snapkv_passkey.py \
  --output-prefix snapkv_passkey_3b_8k
```

The fixed L2 experiment evaluates `no_compression`, `low_l2_keep10`,
`random_keep10`, and `high_l2_keep10` on seeds 0, 1, and 2. Every compressed
configuration keeps 10% of each non-skipped cache layer:

```text
3 seeds x 4 configurations = at most 12 runs
```

The effective number can be lower because a failed baseline skips the three
compressed configurations for that seed. The SnapKV runner evaluates only
`no_compression` and `snapkv`, with a 16-token observation window and a
1024-token target cache by default.

Each prompt is built once per context length and seed. The uncompressed
baseline runs first. If it fails, that baseline row is saved and compressed
configurations are not executed for the case.

For later experiments, change only the context length and output prefix:

```bash
python scripts/run_l2_passkey.py \
  --context-lengths 16384 \
  --output-prefix l2_passkey_3b_16k_keep10

python scripts/run_snapkv_passkey.py \
  --context-lengths 16384 \
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
model_name, method, config, context_length, seed, actual_depth,
target, prediction, correct, target_cache_tokens,
memory_saved_percent, elapsed_seconds
```

The summary contains:

```text
config, context_length, num_examples, accuracy,
mean_memory_saved_percent, mean_elapsed_seconds
```

Torch and Transformers versions, dtype, device map, skip layers, seeds, and
method parameters are stored once in
`results/<output-prefix>_metadata.json`.

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

The existing online LM benchmark is unchanged:

```bash
python scripts/run_online_lm.py
```

## Main files

- `src/l2kv/passkey.py`: exact professor-style prompt construction.
- `src/l2kv/retrieval_eval.py`: shared prefill, compression evaluation, exact
  answer generation, cache measurement, checkpointing, and summaries.
- `scripts/run_l2_passkey.py`: L2-only runner.
- `scripts/run_snapkv_passkey.py`: baseline and SnapKV runner.
- `scripts/plot_retrieval.py`: generic accuracy-versus-context plot.
- `src/l2kv/cache_compression.py`: L2/random cache policies.
- `src/l2kv/snapkv.py`: SnapKV scoring and cache rewrite.

Implementation details are in
[`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md) and
[`docs/SNAPKV_NOTES.md`](docs/SNAPKV_NOTES.md).

## Scope

Compression happens after the full prompt prefill. It reduces the retained
cache and autoregressive attention cost, but not prefill compute or peak prompt
memory. Results are an educational benchmark, not a comprehensive reproduction
of either reference repository.
