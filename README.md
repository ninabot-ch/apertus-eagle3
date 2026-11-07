# apertus-eagle3

Run EAGLE-3 speculative decoding on **Apertus 1.5 70B** with vLLM: how to patch vLLM, a draft head, and a small
script to measure acceptance and throughput.

- Draft head (Hugging Face, gated): `ninabot/Apertus-v1.5-70B-EAGLE3` (see its model card for results and license).
- This repository: instructions, the patch helper, the measurement script. No Apertus or vLLM code or weights.

## Versions this was run with

| Component | Version |
|---|---|
| Container image | `ghcr.io/swiss-ai/vllm_apertus_1.5_release` (arm64 `sha256:8eb36e284fbefdd44c6f4dc23072eff9c06c1caed5643ce2f99ac7e0915a06ae`, amd64 `sha256:d85533f251c8272218ae2e519510836376e5a335b44f2204f193018e37753bc6`) |
| vLLM | 0.23.1rc1 (`0.23.1rc1.dev1029+ga601a9d99`, as shipped in that image) |
| Target model | `swiss-ai/Apertus-v1.5-70B` or a fine-tune of it |
| Draft head | `ninabot/Apertus-v1.5-70B-EAGLE3`, `num_speculative_tokens` = 3 |

The patches target exactly that vLLM. With another vLLM version, read the patches and re-apply by hand; part of
this may already be fixed upstream.

## Why patches

Without them vLLM refuses the head on Apertus 1.5 (`Model instance must have 'model' attribute`). Three small fixes:

| Patch | File in vLLM | What it fixes |
|---|---|---|
| `vllm-apertus-eagle3-aux-layers.patch` | `vllm/model_executor/models/interfaces.py` | `SupportsEagle3` helpers dereference `.model` on what `get_language_model()` returns, which for Apertus 1.5 is already the inner transformer |
| `vllm-apertus-eagle-mrv2-inner.patch` | `vllm/v1/worker/gpu/spec_decode/eagle/utils.py` | same dereference in the model runner v2 EAGLE loader |
| `vllm-apertus-image-token.patch` | `vllm/v1/spec_decode/llm_base_proposer.py` | Apertus 1.5 exposes `image_token_id`, not `image_token_index` |

**These patches are written by the Swiss AI team** and live in
[swiss-ai/apertus-speculative-decoding](https://github.com/swiss-ai/apertus-speculative-decoding) under
`serving/patches/`. We do not copy them; `apply-patches.sh` downloads them at a pinned commit, checks their sha256
and applies them.

## Apply

```bash
# inside the container, or on a vLLM source tree of the same version
./apply-patches.sh /workspace/vllm      # the directory that contains vllm/
```

In a container, bind-mount the patched files over the originals or build a derived image.

## Serve

```bash
vllm serve <apertus-1.5-70b-or-fine-tune> \
  --tensor-parallel-size 4 \
  --speculative-config '{"method":"eagle3","model":"<path to the draft head>","num_speculative_tokens":3}'
```

## Measure

```bash
python3 eagle_diag_bench.py --url http://127.0.0.1:8000 --model <served-name> \
  --generic prompts.jsonl --conc 1,8 --out with-eagle.json
```

Run it once against a server with EAGLE and once without, same prompts. Output: tokens/s and mean acceptance
length per family and concurrency. Greedy decoding, 16 requests at concurrency 1 and 32 at concurrency 8, so
the numbers are indicative, not a benchmark.

## Limits

- The head was trained on a fine-tune of Apertus 1.5 70B, not on the base model. Expect lower acceptance on the
  base (about 1.5 mean acceptance length measured on the base, against about 2 on the fine-tune).
- Tested with W4A16 quantization of the target and tensor parallelism 4. Other quantizations and parallelism
  layouts are untested.
- Small measurement samples, one run per setting.
- The head is a derivative of Apertus and is under the Apertus license and Acceptable Use Policy (Hugging Face
  repository).

## License

Apache-2.0 for the files in this repository (see `LICENSE`, `NOTICE`).
