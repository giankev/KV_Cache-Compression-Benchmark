# SnapKV implementation notes

## Core idea

SnapKV uses the end of a complete prompt to predict which earlier prompt
positions will matter during generation. Split the prompt into a prefix and an
observation window:

```text
prompt = prefix + observation_window
L_prompt = L_prefix + L_obs
```

The prefix is prefilled normally. The observation-window tokens are then
processed before generation while their attention weights toward the prefix are
collected. For an attention head `h` and prefix position `j`, the paper's vote
is:

```text
score[h, j] = sum over observation queries i of attention[h, i, j]
```

These are softmax-normalized attention weights. The prefix columns are selected
after softmax; they are not renormalized after the observation-window columns
are removed.

A one-dimensional pooling operation is applied along the prefix-time axis. The
pooled scores are used only to choose positions: key and value vectors are not
pooled or averaged. Top-k indices are gathered from both K and V, and the full
observation window is appended without compression:

```text
pooled_score = pool1d(score)
indices = topk(pooled_score, k_prefix)
new_K = concat(prefix_K[indices], observation_K)
new_V = concat(prefix_V[indices], observation_V)
```

The implementation restores the selected prefix indices to chronological order
before gathering. Listing 1 in the paper leaves them in score order, so this is
a deliberate stability and readability choice in this repository.

## Pooling

Pooling spreads a strong vote to nearby candidate positions and therefore
encourages the retained tokens to form small contextual clusters. This helps
preserve details around a highly attended token. It does not mean retaining
`kernel_size` tokens for every top-k result; the final prefix budget remains
exactly `k_prefix`.

The implementation supports max and average pooling and defaults to max
pooling. It uses stride 1 and symmetric padding `kernel_size // 2`, as in
Listing 1. An odd kernel is required so that the pooled score has exactly
`L_prefix` positions. The paper reports that max and average pooling performed
similarly, while its main retrieval ablation used max pooling with kernel size
5.

## Grouped-Query Attention adaptation

Qwen2.5 uses more query heads than KV heads. Let:

```text
group_size = num_attention_heads // num_key_value_heads
```

The implementation validates that this division is exact. Eager attention
weights have shape:

```text
[batch, H_query, L_obs, L_prefix]
```

They are reshaped to:

```text
[batch, H_kv, group_size, L_obs, L_prefix]
```

and aggregated over `group_size` and `L_obs`, producing one vote vector per
native KV head:

```text
[batch, H_kv, L_prefix]
```

Contiguous query-head groups vote for their shared KV head. For example, with
four query heads and two KV heads, query heads 0-1 vote for KV head 0 and query
heads 2-3 vote for KV head 1. Sum and mean aggregation differ only by a positive
constant when every group and window has the same size, so they produce the
same top-k ranking.

The cache remains at `H_kv`; it is not permanently repeated to `H_query`. This
preserves the memory benefit of Grouped-Query Attention. The paper's pseudocode
uses a single generic head dimension and does not specify this aggregation, so
the native-GQA behavior is a repository adaptation.

### Sharded models

With `device_map="auto"`, Hugging Face Accelerate can return a layer's attention
scores on a different GPU from that layer's `DynamicCache` tensors. Immediately
before pooling, SnapKV moves only that layer's float32 score tensor to the local
K/V device. Pooling, top-k selection, and gathering then remain local to that
GPU; keys, values, and complete cache layers are never transferred between
devices. The temporary local score copy and selection tensors are released as
soon as the layer has been rewritten.

## Capacity and fair comparison

The compression function accepts a total retained prompt capacity. When that
capacity is derived from a keep ratio, the relation is:

```text
target_capacity = floor(L_prompt * keep_ratio)
k_prefix = target_capacity - L_obs
```

Every compressed layer therefore contains exactly:

```text
k_prefix selected prefix tokens + L_obs observation tokens
```

`target_capacity` must be at least `L_obs`. A keep ratio of 1 is a no-op. Layers
listed in `skip_layers` are not changed; the benchmark uses layers 0 and 1 to
match the existing L2 configurations. The passkey runner supplies one absolute
capacity (1024 tokens by default), rather than evaluating a ratio grid.

This total-capacity rule follows the capacity semantics of Listing 1, where the
observation window is included in `max_capacity_prompt`. The paper's separate
ratio formula defines `floor(p * L_prefix)` selected prefix tokens, which would
give a slightly different total. The repository rule is intentional: an
absolute target describes the complete physical prompt-cache length in
compressed layers. The paper does not prescribe the `(0, 1)` skip-layer
setting.

## SnapKV versus L2 selection

SnapKV is query-aware. Its scores depend on how the final prompt tokens attend
to the prefix, so changing the question can change the retained positions. The
paper's experiments explicitly observe this instruction-dependent behavior.

The L2 strategies are query-independent. They score cached keys using only
their norms and can run after a cache has been built without access to attention
weights. SnapKV instead needs an observation pass and therefore lives in a
separate module rather than being another norm-only option in
`cache_compression.py`.

## Passkey benchmark

SnapKV and the L2 methods receive the same professor-style prompt: one passkey
appears twice inside repeated irrelevant text, at a position selected from the
seed. The complete final question is the prompt suffix and must fit inside the
observation window. Accuracy is a direct token-ID comparison over exactly the
tokenized answer length.

## Scope and limitations

SnapKV compresses the prompt cache only after the prompt has been processed. It
reduces memory and attention work during autoregressive generation, but it does
not reduce prompt-prefill compute or the peak memory needed to read the full
prompt. A prompt that cannot be processed by the base model still cannot be
rescued by this method, and SnapKV does not improve an inherently weak
long-context model.

Generated KV states continue to be appended during decoding; the retained
prompt portion is fixed rather than reselected at every generation step.

This repository implements a small, testable, memory-conscious adaptation for
Hugging Face `transformers==4.57.6` and native GQA caches. It processes the
observation window token by token with eager attention, aggregates each token's
votes immediately, uses explicit logical positions after compression, and
supports heterogeneous layer lengths. The direct tensor rewrite is deliberately
limited to plain full-attention `DynamicCache` layers; sliding, static,
quantized, and offloaded-cache variants are outside this implementation's
scope. It is not a bit-for-bit reproduction of the original Transformers 4.37
monkey patches or all serving optimizations from the official SnapKV code.
