"""Minimal conditioned video generation Space."""

import gc
import os
import random
import tempfile
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import gradio as gr
import imageio.v2 as imageio
import numpy as np
import spaces
import torch
from PIL import Image, ImageOps
from diffusers import AutoencoderKLWan, WanVACEPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
TARGET_WIDTH = 832
TARGET_HEIGHT = 480
OUTPUT_FPS = 16
MAX_FRAMES = 81
MAX_SEED = 2**31 - 1

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)

print(f"[vc-experiment] Loading {MODEL_ID}...", flush=True)
vae = AutoencoderKLWan.from_pretrained(
    MODEL_ID,
    subfolder="vae",
    torch_dtype=torch.float32,
)
pipe = WanVACEPipeline.from_pretrained(
    MODEL_ID,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=3.0,
)
pipe.vae.enable_tiling()
pipe.to("cuda")
print("[vc-experiment] Model ready.", flush=True)


def _to_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] >= 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _resize_control_frame(frame: Image.Image) -> Image.Image:
    return ImageOps.fit(
        frame.convert("RGB"),
        (TARGET_WIDTH, TARGET_HEIGHT),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )


def load_control_video(path: str) -> tuple[list[Image.Image], float]:
    if not path:
        raise gr.Error("Upload a control video first.")

    source = Path(path)
    if not source.exists():
        raise gr.Error("The uploaded video is no longer available. Please upload it again.")

    reader = None
    try:
        reader = imageio.get_reader(str(source))
        metadata = reader.get_meta_data() or {}
        fps = float(metadata.get("fps") or OUTPUT_FPS)
        frames: list[Image.Image] = []

        for index, frame in enumerate(reader):
            if index >= MAX_FRAMES + 1:
                break
            frame = _to_rgb_uint8(np.asarray(frame))
            frames.append(_resize_control_frame(Image.fromarray(frame)))
    except Exception as exc:
        raise gr.Error(f"Could not read the control video: {exc}") from exc
    finally:
        if reader is not None:
            reader.close()

    if not frames:
        raise gr.Error("The control video contains no readable frames.")
    if len(frames) > MAX_FRAMES:
        raise gr.Error(
            f"This Space accepts at most {MAX_FRAMES} frames. "
            "Prepare a shorter input clip."
        )
    if len(frames) < 5:
        raise gr.Error("The control video is too short. Use at least 5 frames.")
    if (len(frames) - 1) % 4 != 0:
        raise gr.Error(
            f"The model requires 4n+1 frames; received {len(frames)}. "
            "Use 5, 9, 13, ..., 81 frames."
        )

    return frames, fps


def _gpu_duration(control_video, prompt, steps, guidance, control_scale, seed, negative_prompt) -> int:
    del control_video, prompt, guidance, control_scale, seed, negative_prompt
    steps = int(steps)
    return int(min(300, max(90, 30 + steps * 4)))


@spaces.GPU(duration=_gpu_duration)
def generate(
    control_video: str,
    prompt: str,
    steps: int,
    guidance: float,
    control_scale: float,
    seed: int,
    negative_prompt: str,
):
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt cannot be empty.")

    frames, input_fps = load_control_video(control_video)
    num_frames = len(frames)
    seed = int(seed)
    if seed < 0:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    try:
        result = pipe(
            video=frames,
            mask=None,
            prompt=prompt.strip(),
            negative_prompt=(negative_prompt or "").strip() or None,
            height=TARGET_HEIGHT,
            width=TARGET_WIDTH,
            num_frames=num_frames,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            conditioning_scale=float(control_scale),
            generator=generator,
        ).frames[0]

        with tempfile.NamedTemporaryFile(prefix="vc_experiment_", suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name
        export_to_video(result, output_path, fps=OUTPUT_FPS)

        status = (
            f"Generated {num_frames} frames at {TARGET_WIDTH}×{TARGET_HEIGHT}, {OUTPUT_FPS} fps. "
            f"Input fps: {input_fps:.2f}. Seed: {seed}."
        )
        return output_path, status
    except gr.Error:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        raise gr.Error(
            "GPU memory was exhausted. Try fewer frames or fewer inference steps."
        ) from exc
    except Exception as exc:
        raise gr.Error(f"Generation failed: {type(exc).__name__}: {exc}") from exc
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


with gr.Blocks(title="VC Experiment") as demo:
    gr.Markdown(
        "# VC Experiment\n"
        "Upload a control video, enter a text prompt, and generate a video."
    )

    with gr.Row():
        with gr.Column(scale=1):
            control_video = gr.Video(
                label="Control video",
                sources=["upload"],
                format="mp4",
            )
            prompt = gr.Textbox(
                label="Prompt",
                lines=4,
                placeholder="Describe the desired output...",
            )

            with gr.Accordion("Inference settings", open=False):
                steps = gr.Slider(10, 40, value=20, step=1, label="Inference steps")
                guidance = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Guidance scale")
                control_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Control scale")
                seed = gr.Number(value=42, precision=0, label="Seed (-1 = random)")
                negative_prompt = gr.Textbox(
                    value=DEFAULT_NEGATIVE_PROMPT,
                    label="Negative prompt",
                    lines=4,
                )

            generate_button = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            result_video = gr.Video(label="Result", autoplay=True)
            status = gr.Markdown()

    generate_button.click(
        fn=generate,
        inputs=[control_video, prompt, steps, guidance, control_scale, seed, negative_prompt],
        outputs=[result_video, status],
    )

    gr.Markdown(
        "**Input:** 5–81 frames; frame count must be `4n+1`. "
        "Frames are normalized to 832×480."
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
