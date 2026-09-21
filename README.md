---
title: StoneRiver VACE Depth
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
python_version: "3.12"
startup_duration_timeout: 1h
models:
  - Wan-AI/Wan2.1-VACE-1.3B-diffusers
---

# StoneRiver VACE Depth

Minimal Hugging Face Space for the StoneRiver experiment:

```text
depth.mp4 + text prompt
        ↓
Wan 2.1 / VACE 1.3B
        ↓
result.mp4
```

This is **video-to-video with a depth control sequence**. It is not plain text-to-video and it does not use only a reference image. The Space expects the depth video to be prepared beforehand and never runs a depth estimator.

## Why this really is depth-controlled V2V

The implementation follows the official VACE/Wan behavior rather than treating the uploaded MP4 as a generic reference clip.

In the official VACE user guide, depth is a `control` task and the preprocessed depth sequence is passed as `src_video`. The official Wan2.1 `prepare_source` implementation sets `src_mask = ones_like(src_video)` when a source video is provided without a mask. Then `vace_encode_frames` puts the source video into the reactive/control channels, `vace_latent` builds the VACE control context, and that context is passed to the model on every denoising step. The Diffusers `WanVACEPipeline` mirrors this: when `video` is provided and `mask=None`, it creates an all-ones mask and encodes the whole video as VACE conditioning latents. The VACE transformer injects those control hints into its configured VACE layers throughout denoising.

Relevant upstream sources:

- https://github.com/ali-vilab/VACE
- https://github.com/ali-vilab/VACE/blob/main/UserGuide.md
- https://github.com/Wan-Video/Wan2.1/blob/main/wan/vace.py
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_vace.py
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan_vace.py

`app.py` therefore deliberately calls `WanVACEPipeline(video=depth_frames, mask=None, ...)`. No reference images, normals, masks, pose, or other controls are enabled in this first version.

## Model and ZeroGPU choice

The Space uses `Wan-AI/Wan2.1-VACE-1.3B-diffusers` at 832×480. The official Wan repository supports VACE 1.3B for 480p and recommends 480p over 720p for the small model. The 14B Diffusers checkpoint is about 75 GB and is unnecessary for this first ZeroGPU experiment; the 1.3B Diffusers checkpoint is about 19 GB and is already demonstrated on Hugging Face ZeroGPU.

ZeroGPU currently provides a 48 GB `large` GPU slice by default and supports Gradio + the `@spaces.GPU` decorator. Model loading is done at module scope and the generation function is decorated with `@spaces.GPU`, following Hugging Face's recommended pattern.

## Input constraints

This intentionally small first version accepts 5–81 frames and requires the frame count to be `4n+1`, matching Wan/VACE temporal requirements. Each uploaded depth frame is preserved in order and resized/cropped to 832×480. Output is encoded at 16 fps, matching the official Wan/VACE sampling convention.

For an initial smoke test, use a 49-frame depth clip and the defaults below. Once that is stable, use the standard 81-frame clip.

Default inference parameters:

- 20 inference steps
- guidance scale 5.0
- depth conditioning scale 1.0
- seed 42
- UniPC scheduler, `flow_shift=3.0` for 480p
- 16 fps output

## Create the Hugging Face Space once

1. On Hugging Face, create a new **Gradio** Space. A practical name is `StoneRiverVACE`.
2. Choose **ZeroGPU** hardware in the Space settings. A free personal account in good standing can currently host up to two ZeroGPU Spaces; otherwise use the hardware available to your HF account.
3. The Space can start empty. Do not manually copy the code; GitHub Actions will push this repository to it.
4. Create a Hugging Face user access token with **write** access to that Space/repository.

If the model download makes the first boot slow, keep `startup_duration_timeout: 1h` in this README. The model is public and does not require an HF token for inference downloads.

## Configure GitHub once

In `StoneRiverVACE` → **Settings → Secrets and variables → Actions** add:

### Secret

- `HF_TOKEN` — Hugging Face token with write permission to the target Space.

### Variables

- `HF_USERNAME` — the Hugging Face username that owns the token.
- `HF_SPACE_ID` — full Space id, for example `your-hf-user/StoneRiverVACE`.

The workflow is `.github/workflows/sync-to-huggingface.yml`. Every push to GitHub `main` force-pushes the same commit history to the Space `main` branch. Hugging Face automatically rebuilds/restarts the Space after the push.

You can also trigger the workflow manually from **GitHub → Actions → Sync to Hugging Face Space → Run workflow**.

## What must be done manually exactly once

- Create the HF Space and select ZeroGPU hardware.
- Create the HF write token.
- Add `HF_TOKEN`, `HF_USERNAME`, and `HF_SPACE_ID` to GitHub Actions settings.
- Run the workflow once manually if you do not want to wait for the next push.

After that, normal development is only GitHub commits/pushes to `main`; deployment to Hugging Face is automatic.

## Local run (optional)

A local CUDA machine can run the same app. Install a CUDA-enabled PyTorch version compatible with your system, then:

```bash
pip install -r requirements.txt
python app.py
```

The model is downloaded from Hugging Face on first start, so expect substantial disk use and startup time.
