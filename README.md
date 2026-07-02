# KV Cache Compression Benchmark

Minimal L2-norm KV-cache compression benchmark for Qwen2.5-3B-Instruct.

The main benchmark path is intentionally small:

- standard passkey task only
- context lengths: `8192`, `32768`
- depth: `0.5`
- seed: `0`
- configs: `no_compression`, `low_l2_keep50`, `low_l2_keep10`
- compressed configs use `get_default_skip_layers()`

## Kaggle Usage

```bash
pip install -e .
python scripts/run_basic_passkey.py
```

The benchmark prints raw and summary dataframes, then saves:

- `results/basic_passkey_raw.csv`
- `results/basic_passkey_summary.csv`

For the token-by-token WikiText online LM evaluation:

```bash
python scripts/run_online_lm.py
```

This writes `results/online_lm_summary.csv`.

To save an attention/L2 heatmap:

```bash
python scripts/show_attention_l2.py
```

This writes `results/attention_l2_heatmap.png`. Add `--show` to display it.

`src/l2kv/alr.py` and `scripts/run_alr_scan.py` are kept for exploratory ALR
work, but they are not part of the main benchmark path.

## Structure

- `scripts/run_basic_passkey.py` - the main benchmark.
- `scripts/run_online_lm.py` - token-by-token WikiText online LM benchmark.
- `scripts/show_attention_l2.py` - attention/L2 visualization helper.
- `scripts/show_alr_heatmap.py` - layer/head ALR heatmap helper.
- `scripts/run_alr_scan.py` - optional ALR scan.
- `src/l2kv/cache_compression.py` - in-place DynamicCache compression.
- `src/l2kv/cache_metrics.py` - actual and theoretical KV cache sizes.
- `src/l2kv/configs.py` - the three basic benchmark configs.
- `src/l2kv/model_utils.py` - model/tokenizer loading.
- `src/l2kv/passkey.py` - standard passkey prompt and strict-context generation.
- `src/l2kv/attention_viz.py` - attention/L2 and ALR heatmaps.
- `src/l2kv/alr.py` - ALR scan and skip-layer suggestions.
