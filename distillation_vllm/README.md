# HMM Distillation with vLLM (text **and** vision-language models)

This directory provides a **vLLM-accelerated** reimplementation of the HMM
distillation pipeline in [`../distillation`](../distillation). It replaces the
original Hugging Face `model.generate()` sampling loop with an OpenAI-compatible
vLLM server, giving roughly **~10× faster** output sampling, and adds support for
distilling HMMs from **vision-language models (VLMs)** in addition to text LMs.

The trained HMM can be used as a tractable proxy for the base model exactly as in
the main [Ctrl-G](../README.md) tutorial (constrained generation via DFA).

## What's different from `distillation/`

| | `distillation/` (upstream) | `distillation_vllm/` (this dir) |
|---|---|---|
| Output sampling | `transformers.generate()` | vLLM OpenAI server (`~10×` faster) |
| Modality | Text LMs | Text LMs **and** VLMs |
| LVD embeddings | single process | multi-GPU via `torchrun` |
| Constrained gen | — | `ctrlg` logits processor patched for vLLM (variable-length batches, preemption-resumed prefixes, cache cleanup, `alpha` weighting — see [`../ctrlg/utils.py`](../ctrlg/utils.py)) |

## Requirements

In addition to the base Ctrl-G requirements (`pip install -e ..`):

```bash
pip install vllm openai faiss-gpu
```

A multi-GPU machine is recommended. Sampling runs on a vLLM server; LVD embedding
extraction and EM training run separately via `torchrun`.

> **Note on data.** The large prompt/response datasets used in our experiments
> (DAPO-Math, OpenVLThinker) are **not** included. The tutorial notebooks build
> the sampling inputs directly from the public Hugging Face datasets
> (`open-r1/DAPO-Math-17k-Processed`, `ydeng9/OpenVLThinker-grpo-*`). See the
> [Input data format](#input-data-format) section to plug in your own data.

## Pipeline overview

The pipeline has four stages, identical for both modalities:

1. **Prepare** the sampling prompts (a JSON list of prompts; for VLMs each entry
   also carries a base64 image).
2. **Sample** outputs from the base model through a vLLM server.
3. **Initialize** HMM `checkpoint-0` via Latent Variable Distillation (LVD).
4. **Train** the HMM with Expectation-Maximization (EM).

### Tutorials (start here)

- **Text LM:** [`train_hmm.ipynb`](train_hmm.ipynb) — distills an HMM from a
  Qwen-family base model on DAPO-Math.
- **Vision-language:** [`train_hmm_for_vlm.ipynb`](train_hmm_for_vlm.ipynb) —
  distills an HMM from a Qwen2.5-VL / OpenVLThinker base model on OpenVLThinker
  data.

Each notebook prints the exact shell commands for every stage; run those commands
in a terminal (the vLLM server in particular must run as its own process).

## Files

**Shared (both modalities)**
- [`sample_data_vllm.sh`](sample_data_vllm.sh) — driver that produces the `dev` /
  `lvd` / `train` splits by calling the right sampler (`IS_VLM` toggles between
  text and image samplers).
- [`lvd_hmm.py`](lvd_hmm.py) — k-means over LVD embeddings to build HMM
  `checkpoint-0`.
- [`train_hmm.py`](train_hmm.py) — multi-GPU EM training (`torchrun`).

**Text LM track**
- [`launch_vllm_model.sh`](launch_vllm_model.sh) — start the vLLM server.
- [`sample_data_instruct_vllm.py`](sample_data_instruct_vllm.py) — text sampler.
- [`get_lvd_embedding.py`](get_lvd_embedding.py) /
  [`run_get_lvd_embedding.sh`](run_get_lvd_embedding.sh) — extract fp32 hidden
  states for LVD (multi-GPU).

**Vision-language track**
- [`launch_vllm_model_vlm.sh`](launch_vllm_model_vlm.sh) — start the vLLM server
  with `min/max_pixels` image config.
- [`sample_data_instruct_vllm_img.py`](sample_data_instruct_vllm_img.py) — image
  sampler.
- [`get_lvd_embedding_img.py`](get_lvd_embedding_img.py) /
  [`run_get_lvd_embedding_img.sh`](run_get_lvd_embedding_img.sh) — VLM LVD
  embeddings.

**Data prep & evaluation**
- [`preproc_dapo_to_json.py`](preproc_dapo_to_json.py),
  [`preproc_dapo_boxed_to_json.py`](preproc_dapo_boxed_to_json.py) — convert
  DAPO-Math into the sampling-input JSON format.
- [`run_eval.ipynb`](run_eval.ipynb), [`run_eval_vlm.ipynb`](run_eval_vlm.ipynb)
  — evaluate a trained HMM against the base model.

## Quick start (text LM)

```bash
# 0. Build sampling prompts (see train_hmm.ipynb Step 1), e.g. dapo_prompts.json

# 1. Launch the vLLM server (own terminal). Args: GPUS  MODEL
./launch_vllm_model.sh 0,1,2,3 Qwen/Qwen3-1.7B-Base

# 2. Sample dev/lvd/train splits. Args: MODEL  NAME  INPUT_JSON  OUTPUT_DIR  IS_VLM
./sample_data_vllm.sh Qwen/Qwen3-1.7B-Base my_model dapo_prompts.json ./out False

# 3. Extract LVD embeddings (fp32). Args: LVD_FILE  MODEL  GPUS
./run_get_lvd_embedding.sh ./out/my_model.lvd Qwen/Qwen3-1.7B-Base 0,1,2,3

# 4. Initialize checkpoint-0 (lvd_hmm.py) and train (train_hmm.py)
#    — see train_hmm.ipynb Steps 3-4 for the full commands.
```

For VLMs, use `launch_vllm_model_vlm.sh` / `run_get_lvd_embedding_img.sh` (both
take additional `MIN_PIXELS MAX_PIXELS` args) and pass `IS_VLM=True` to
`sample_data_vllm.sh`. See [`train_hmm_for_vlm.ipynb`](train_hmm_for_vlm.ipynb).

## Input data format

`sample_data_vllm.sh` consumes a JSON file containing a list of prompts.

- **Text LM** — a list of strings:
  ```json
  ["Question 1 ...\nPlease reason step by step ...\n", "..."]
  ```
- **VLM** — a list of objects, each with a base64-encoded image:
  ```json
  [{"prompt": "...", "img_b64": "<base64>", "mime_type": "image/png"}, ...]
  ```

The sampler writes three splits next to `OUTPUT_DIR`: `.dev`, `.lvd`
(**do not shuffle the LVD split**), and `.train` (chunked).

## Notes & tips

- The vLLM server serves fp16; LVD embedding extraction reloads the model in
  fp32 to recover hidden states accurately.
- `alpha` (in the `ctrlg` logits processor) scales the HMM's influence during
  constrained generation — see [`../ctrlg/utils.py`](../ctrlg/utils.py).
- Example model paths in the scripts (`/path/to/your/model_or_checkpoint`) are
  placeholders — replace with a Hugging Face id or a local checkpoint path.

## Acknowledgements

This builds directly on [Ctrl-G](https://github.com/joshuacnf/Ctrl-G)
(Zhang et al.). vLLM acceleration and VLM support were added by the contributors
of this directory. Sampling is powered by [vLLM](https://github.com/vllm-project/vllm).
