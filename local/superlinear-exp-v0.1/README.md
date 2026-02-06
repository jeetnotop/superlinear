---
license: other
license_name: nvidia-open-model-license
license_link: LICENSE
library_name: transformers
pipeline_tag: text-generation
tags:
  - long-context
  - superlinear-attention
  - subquadratic
  - causal-lm
base_model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
---

# Superlinear-Exp-v0.1

**Superlinear Multi-Step Attention** — a subquadratic attention mechanism that preserves random context access (structural non-exclusion) for extremely long sequences.

This is an experimental release demonstrating the Superlinear attention architecture integrated into a modified [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) hybrid model.

> **WARNING (Security):** This model requires `trust_remote_code=True`, which executes Python code from this repository. Review the code before running in sensitive environments.

## Model Description

Superlinear attention reformulates causal self-attention as a multi-step search problem:

1. **Accumulation** — Efficiently processes the sequence and produces per-position representatives (via Mamba-2 layers in the hybrid architecture).
2. **Span Search** — Scores a sublinear number of candidate spans using learned routing, then selects top-k spans per query.
3. **Span Attention** — Computes standard token-level attention within the selected contiguous spans.
4. **Combination** — Produces outputs using softmax-weighted gating over span attention outputs.

In the baseline N=2 implementation, both span-search and span-attention scale as **O(L^(3/2))**, enabling practical inference at multi-million-token context lengths where dense attention becomes prohibitive.

**Key property:** *Random context access* (structural non-exclusion) — any eligible token position can be selected by the content-dependent routing mechanism; no fixed sparsity pattern permanently excludes positions.

## Quickstart

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "concavity-ai/superlinear-exp-v0.1",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "concavity-ai/superlinear-exp-v0.1",
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)

messages = [{"role": "user", "content": "Explain the Transformer architecture."}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

output = model.generate(inputs, max_new_tokens=1000, do_sample=True, temperature=0.1, top_p=0.99)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Dependencies

This model uses custom Python code (`trust_remote_code=True`) and CUDA extensions.

### Recommended: follow the Superlinear repo install

The simplest supported path is to use the installation flow from the Superlinear repo (it pins a known-good CUDA toolchain and builds `mamba-ssm[causal-conv1d]` from source to avoid wheel/ABI mismatches):

https://github.com/concavity-ai/superlinear#installation

Copy/paste one-liner (from the Superlinear repo root):

```bash
conda env create -f environment.yml \
  && conda run -n superlinear pip install torch --index-url https://download.pytorch.org/whl/cu128 \
  && conda run -n superlinear pip install -e ".[server,model]" \
  && conda run -n superlinear bash -lc 'CUDA_HOME="$CONDA_PREFIX" pip install "mamba-ssm[causal-conv1d]" --no-build-isolation --no-cache-dir --no-binary :all:'
```

### Optional: pip-only (if `mamba-ssm` already works in your env)

If `python -c "import mamba_ssm, causal_conv1d"` already succeeds in the environment you’ll run inference in, you already have a working PyTorch/CUDA pairing for the extension in that environment — you should not need to reinstall PyTorch.

Install the remaining Python deps + Superlinear:

```bash
pip install -U "transformers<5" accelerate safetensors
pip install -U vllm triton

# Superlinear kernels (span-attention)
pip install -U git+https://github.com/concavity-ai/superlinear.git
```

### Building `mamba-ssm` from source (only if needed)

If you must build `mamba-ssm[causal-conv1d]` yourself, you need a CUDA toolkit with `nvcc` and `CUDA_HOME` pointing at it (example: `/usr/local/cuda`):

```bash
CUDA_HOME=/usr/local/cuda \
  pip install -U "mamba-ssm[causal-conv1d]" \
  --no-build-isolation --no-cache-dir --no-binary :all:
```

## Recommended Inference Settings

For long-context inference with the Superlinear attention mechanism, use the following configuration:

```python
model = AutoModelForCausalLM.from_pretrained(
    "concavity-ai/superlinear-exp-v0.1",
    # Attention implementation
    _attn_implementation='block-span-gqa',
    decode_kernel='staged-gqa',
    
    # Performance optimizations
    enable_cuda_graph=True,
    enable_shared_fused_moe=True,
    
    # Superlinear attention hyperparameters
    span_attention_sw_index=65,           # Local window boundary index
    span_attention_num_spans=3,           # Top-k spans per query
    span_attention_backward_factor=3,     # Backward span extent multiplier
    span_attention_forward_factor=1,      # Forward span extent multiplier
    span_attention_search_power=0.55,     # Search exponent (controls anchor budget)
    span_attention_span_power=0.55,       # Span exponent (controls span scale)
    
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)
```

### Hyperparameter Notes

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `span_attention_num_spans` | Number of routed spans selected per query (top-k) | 2 or 3 |
| `span_attention_backward_factor` | Backward extent of each span relative to base scale | 2–4 |
| `span_attention_forward_factor` | Forward extent of each span relative to base scale | 0–2 |
| `span_attention_search_power` | Exponent controlling the number of candidate anchors | 0.5–0.667 |
| `span_attention_span_power` | Exponent controlling span length scaling | 0.5–0.667 |

**Sliding window length from `span_attention_sw_index`:** Internally, the kernels compute the sliding-window length as:

```text
window_len = floor((sw_index + 1) ** (1 / search_power)) - 1
```

We parameterize the local sliding window using `sw_index` (a stride/stripe index) rather than specifying `window_len` directly. This keeps the sliding-window boundary aligned with the same index space used by span search, so span-search begins immediately after the sliding-window region and avoids gaps between local attention and routed spans.

Example: with `search_power=0.55` and `sw_index=65`,

```text
window_len = floor(66 ** (1 / 0.55)) - 1 = 2032
```

## Hardware Requirements

- **GPU:** NVIDIA GPU with sufficient VRAM (tested on B200 180GB)
- **KV Cache:** ~6GB per million tokens (model-dependent)
- **Precision:** FP16 recommended

### Measured Throughput (Single B200, Batch=1)

| Context Length | Prefill (tok/s) | Decode (tok/s) |
|----------------|-----------------|----------------|
| 1M tokens      | ~20,202         | ~109           |
| 10M tokens     | ~5,576          | ~76            |

*Your results may vary depending on hardware, batch size, and configuration.*

## Intended Use

This is an **architecture-and-systems feasibility study** release. It demonstrates that:

1. The Superlinear attention mechanism is structurally random-context-access-preserving
2. The architecture achieves asymptotically subquadratic attention complexity
3. The resulting irregular span pattern can be implemented with practical performance at very long context lengths

### Limitations

- **Not a comprehensive quality study:** We do not present extensive ablations or claim state-of-the-art accuracy on benchmarks.
- **Limited evaluation:** Initial validation focused on NIAH (Needle In A Haystack) retrieval task and throughput measurements.
- **Experimental:** This release is intended for research and experimentation, not production use.
- **Memory:** Full KV cache must be retained for random context access; memory usage scales with context length.

## What's in This Repository

```
├── config.json                     # Model configuration
├── generation_config.json          # Default generation settings
├── tokenizer.json                  # Tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
├── chat_template.jinja             # Chat template
├── configuration_superlinear_exp.py  # Custom config class
├── modeling_superlinear_exp.py     # Custom model implementation
├── moe.py                          # MoE components
├── model-*.safetensors             # Model weights (16 shards)
├── model.safetensors.index.json    # Weight index
├── LICENSE                         # NVIDIA Open Model License
├── NOTICE                          # Required attribution
└── README.md                       # This file
```

## License

### Model Weights

This model is a derivative of [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) and is distributed under the **NVIDIA Open Model License Agreement**.

See [LICENSE](LICENSE) for the full license text.

Use of this model must be consistent with [NVIDIA's Trustworthy AI terms](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/).

### Code

The modeling code in this repository is provided for loading and running the model. For the broader Superlinear project codebase, see [github.com/concavity-ai/superlinear](https://github.com/concavity-ai/superlinear) (Apache-2.0).

## Attribution

**Upstream Model:**
- NVIDIA Nemotron-3-Nano-30B-A3B ([nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16))

**Paper:**
```bibtex
@article{huang2026superlinear,
  title={Superlinear Multi-Step Attention},
  author={Huang, Yufeng},
  journal={arXiv preprint arXiv:2601.18401},
  year={2026}
}
```

## Patent Notice

Patent applications have been filed related to aspects of the methods described in this work.

## Contact

- Author: Yufeng Huang
- Email: yufeng@concavity.ai
- Organization: Concavity AI
