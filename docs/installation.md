# Installation (running the released model)

This guide is for running **concavity-ai/superlinear-exp-v0.1**. The model’s remote code depends on CUDA-compiled extensions (notably `mamba-ssm[causal-conv1d]`) plus Python runtime deps (`transformers`, `accelerate`, `vllm`).

## Recommended (copy/paste): conda + CUDA 12.8 + source-build `mamba-ssm`

Run this from the repo root:

```bash
conda env create -f environment.yml \
	&& conda run -n superlinear pip install torch --index-url https://download.pytorch.org/whl/cu128 \
	&& conda run -n superlinear pip install -e ".[server,model]" \
	&& conda run -n superlinear bash -lc 'CUDA_HOME="$CONDA_PREFIX" pip install "mamba-ssm[causal-conv1d]" --no-build-isolation --no-cache-dir --no-binary :all:'
```

Quick verification:

```bash
conda run -n superlinear python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
conda run -n superlinear python -c "import mamba_ssm, causal_conv1d, vllm; print('mamba_ssm', mamba_ssm.__version__)"
```

## Why this isn’t a simple `pip install` (yet)

In an ideal world, this would be pip-only.

Right now, `mamba-ssm[causal-conv1d]` does not reliably support **CUDA 13.1** environments, so we standardize on a known-good toolchain (**CUDA 12.8 + nvcc**) and build it from source against your installed `torch`.

Once `mamba-ssm[causal-conv1d]` supports CUDA 13.1 (or ships compatible wheels), we plan to simplify this to a pip-only install.

## Step-by-step (same as the one-liner)

```bash
conda env create -f environment.yml
conda activate superlinear

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[server,model]"

CUDA_HOME="$CONDA_PREFIX" \
	pip install "mamba-ssm[causal-conv1d]" \
	--no-build-isolation --no-cache-dir --no-binary :all:
```

## Optional: pip-only if you already have `mamba-ssm` working

If you have **already** successfully installed `mamba-ssm[causal-conv1d]` (and `import mamba_ssm, causal_conv1d` works in the same environment you plan to run the server in), you can skip the conda toolchain steps and just install Superlinear + its Python deps:

```bash
python -c "import mamba_ssm, causal_conv1d; print('mamba_ssm ok')"

# Install Superlinear + server + model Python deps
pip install -e ".[server,model]"
```

If `torch` is not installed yet in that environment, install a CUDA-matching build first (example for cu128):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## CUDA / nvcc compatibility (sanity check)

- **Tested with CUDA 12.8** (`nvcc 12.8`).
- All GPU packages must be compatible with the same CUDA major/minor.

Quick checks:

```bash
nvcc --version | grep release
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
```

If these disagree (e.g., torch reports `CUDA=12.4` but `nvcc` is 12.8), expect build/runtime failures.

## Transformers version

Tested with **transformers v4.57.x**. Transformers **v5** may introduce breaking changes and has not been validated.

If you hit issues, pin v4:

```bash
pip install "transformers>=4.38,<5"
```

## Tested versions (known-good)

| Package | Version |
|---|---|
| nvcc | 12.8.93 |
| torch | 2.9.1+cu128 |
| triton | 3.5.1 |
| transformers | 4.57.6 |
| mamba-ssm | 2.3.0 |
| causal-conv1d | 1.6.0 |
| vllm | 0.15.0 |

