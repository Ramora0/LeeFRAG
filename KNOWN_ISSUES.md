# Known Issues & Potential Problems

## Model Choice

### Data Contamination (glaiveai/Llama-3-8B-RAG-v1)
The frozen LLM was fine-tuned on the same dataset we train the Q-Former on. This means:
- The model may partially predict answers from parametric memory rather than document context.
- Hidden states could carry weaker document-dependent signal than expected.
- The Q-Former might learn that it can discard document information and still achieve low loss.

**Mitigation:** KL divergence loss measures behavioral similarity (full vs compressed context) rather than answer correctness. If the model doesn't need documents, KL is small for both conditions, and the Q-Former receives weak but directionally correct gradients. Still, this is the single biggest risk to training quality.

**Validation:** Run `test.py` once WITH documents and once WITHOUT (pass empty docs). If perplexity is similar, the model doesn't depend on documents and the training signal is fundamentally weak.

### Alternative Models Not Tested
- `meta-llama/Meta-Llama-3-8B-Instruct`: No contamination but instruct models tend to hallucinate rather than ground in documents, producing a different kind of weak signal.
- Base LLaMA 3 8B: Cleanest in-context learning but doesn't understand the chat format or citation tags.
- No well-validated general RAG model exists for Llama 3 8B outside of glaive.

## Gradient Flow

### Gradients Through DynamicCache
The entire training signal for the Q-Former depends on gradients flowing through `DynamicCache` → LLM attention → loss. Potential failure points:
- `DynamicCache.update()` must not call `.detach()` or `.clone()` on the KV tensors. Current HuggingFace transformers (4.40+) preserves the graph, but future versions could break this silently.
- Some HF model implementations may `.contiguous()` or reshape KV tensors in ways that break gradient tracking in edge cases.
- `verify_gradient_flow()` in `train.py` checks this before training starts, but only on one batch.

### Flash Attention Backward Through KV
`F.scaled_dot_product_attention` may not dispatch to the flash attention kernel when KV inputs have `requires_grad=True`. If it falls back to the math implementation:
- Attention matrices are materialized: `[batch, 32_heads, seq_len, prefix+seq_len]` per layer.
- Memory usage jumps by 4-8 GB, potentially causing OOM on 40GB A100.
- No error is raised — it just silently uses more memory.

**Check:** Monitor GPU memory on the first training step. If it's above ~32 GB, flash attention likely isn't dispatching.

## Memory

### Hidden State Storage
Retaining 32 layers of hidden states between Stage A and Stage B costs ~384 MB per sample (at ~1500 doc tokens). This is unavoidable in the current architecture. With longer documents or if `max_total_doc_tokens` is increased, this scales linearly.

### Teacher Logits
Full teacher logits over 128k vocab are ~128 MB per sample. Use `--kl_top_k 1000` to reduce to ~6 MB, at some cost to KL loss fidelity.

### No Batching
`batch_size` is hardcoded to 1 due to variable document counts and lengths per sample. The collator assumes a single item. Increasing batch size would require padding/packing logic that doesn't exist yet.

### Stage A Recomputation
Stage A (frozen LLM forward) is recomputed every epoch for every sample. Over 4 epochs, this is 4x redundant work (~75% of which could theoretically be cached. Disk caching is impractical (~17 TB for full hidden states).

## Architecture

### Shared Input Projection
All 8 Q-Former layers share the same `input_proj` (4096 → 1024) for projecting LLM hidden states. Early and late LLM layers have very different representations, so a shared projection may be suboptimal. Per-group projections would add ~33M params but might improve quality.

### Query Embeddings Are Content-Agnostic
The learned query embeddings are the same regardless of document content. Sinusoidal positional embeddings give position awareness, but the queries have no conditioning on the actual document before cross-attention. A content-dependent initialization (e.g., pooled document features) might help, especially at high compression ratios.

### Compression Schedule Rigidity
The schedule divides training into 4 equal phases (2x, 4x, 8x, 16x). If the model hasn't converged at 2x before switching to 4x, higher compression phases inherit a poor compressor. There's no adaptive mechanism — it switches purely by step count.

### Sequential Q-Former Document Processing
The Q-Former processes documents one at a time in a loop (`_stage_b`), unlike Stage A which concatenates all documents and uses a block-diagonal causal mask for a single batched forward pass. The Q-Former could similarly concatenate all documents' queries and hidden states into one forward pass with block attention masks on both self-attention and cross-attention, keeping documents independent while enabling GPU parallelism. This would also require handling variable `num_queries` per document (due to different doc lengths).

### ~~RoPE Leaks Absolute Position Into Compressed KV~~ (FIXED)
**Fixed:** `apply_rope_to_cache()` in `kv_cache_utils.py` now applies the LLM's RoPE to the Q-Former's output K values at prefix positions `[0, prefix_len)` before Stage B. The Q-Former outputs raw (un-rotated) K/V, then RoPE is applied to K using the model's own `rotary_emb`, matching what the LLM expects from a normal KV cache. V is left as-is (LLaMA never rotates V). Note: the hidden states fed to the Q-Former are residual-stream outputs, not Q/K projections, so they don't carry direct RoPE rotations — only implicit positional signal through attention patterns.

### Single-Document Compression
Each document is compressed independently. Cross-document information (redundancy, contradictions) is not exploited during compression. Two documents saying the same thing produce two separate compressed caches rather than being deduplicated.

## Data

### Glaive Format Quirks
- `<co:N>...</co>` citation tags are non-standard tokens. The tokenizer may split them unpredictably, adding noise to the CE loss on answer tokens.
- `Answer Mode: Grounded` vs `Mixed` changes expected behavior. Mixed mode answers include parametric knowledge, making the document dependency weaker for those samples.
- Some answers in the dataset have 0 characters (empty `answer` field). These produce no CE loss signal and waste a training step.

### No Validation of Document Relevance
The dataset assumes all provided documents are relevant. If some documents are distractors, the Q-Former might learn to compress distractor information, wasting capacity.

### 90/10 Split Is Not Stratified
The train/eval split is a simple shuffle-and-cut. It's not stratified by answer mode, document count, or document length. Eval performance could be unrepresentative if the split is unlucky.

## Training Stability

### KL + CE Scale Mismatch
CE loss and KL divergence can be on very different scales, especially early in training when the compressed representation is poor. The default `kl_weight=1.0` may need tuning. If KL dominates, the Q-Former optimizes for distribution matching but ignores answer correctness. If CE dominates, the KL signal is wasted.

**Watch:** If `train/kl_loss` is 10x+ larger than `train/ce_loss` early in training, reduce `--kl_weight`.

### No Learning Rate per Phase
The cosine LR schedule spans all 4 epochs continuously. When the compression ratio jumps (e.g., 4x → 8x), the model faces a harder task but the learning rate has already decayed. A per-phase warmup or LR reset might help.

### Gradient Accumulation Across Compression Boundaries
If a compression ratio change happens mid-accumulation (e.g., step 5775 of 5775 switches from 2x to 4x), the accumulated gradients mix signals from two different compression levels. This is a minor issue but could cause instability at phase boundaries.


- mean pool init
- ~~Proper ROPE~~ (done)
- No self attention??
- Per layer
- query init