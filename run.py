"""
GeometryCrafter: Video-to-Geometry Generation Script
Optimized for high-precision 3D geometry extraction from video sequences.
"""

from pathlib import Path
import torch
from decord import VideoReader, cpu
from diffusers.training_utils import set_seed
from fire import Fire
import os
import numpy as np
import torch.nn.functional as F

from third_party import MoGe
from geometrycrafter import (
    GeometryCrafterDiffPipeline,
    GeometryCrafterDetermPipeline,
    PMapAutoencoderKLTemporalDecoder,
    UNetSpatioTemporalConditionModelVid2vid
)

def main(
    video_path: str,
    save_folder: str = "workspace/output/",
    cache_dir: str = "workspace/cache", 
    height: int = None,
    width: int = None,
    downsample_ratio: float = 1.0,
    num_inference_steps: int = 5,
    guidance_scale: float = 1.0,
    window_size: int = 110,
    decode_chunk_size: int = 8,
    overlap: int = 25,
    process_length: int = -1,
    process_stride: int = 1,
    seed: int = 42,
    model_type: str = 'diff', # 'determ'
    force_projection: bool = True,
    force_fixed_focal: bool = True,
    use_extract_interp: bool = False,
    track_time: bool = False,
    low_memory_usage: bool = False
):    
    """
    Main entry point for GeometryCrafter inference.

    Args:
        video_path (str): Path to the input video file (mp4, avi, etc.).
        save_folder (str): Directory where the output .npz files will be saved.
        cache_dir (str): Directory to store downloaded model weights.
        height (int): Output video height. Must be a multiple of 64.
        width (int): Output video width. Must be a multiple of 64.
        downsample_ratio (float): Optional factor to shrink input before processing.
        num_inference_steps (int): Number of diffusion steps (default 5).
        guidance_scale (float): Classifier-free guidance scale (usually 1.0).
        window_size (int): Number of frames processed in one temporal window.
        decode_chunk_size (int): Frames processed at once by the VAE.
        overlap (int): Number of overlapping frames between temporal windows.
        process_length (int): Total number of frames to process. -1 for entire video.
        process_stride (int): Skip frames (e.g., 2 processes every other frame).
        seed (int): Random seed for reproducibility.
        model_type (str): 'diff' for diffusion, 'determ' for deterministic (faster).
        force_projection (bool): If True, enforces perspective projection constraints.
        force_fixed_focal (bool): If True, assumes a constant camera focal length.
        use_extract_interp (bool): Use interpolation during feature extraction.
        track_time (bool): Print timing information for each stage.
        low_memory_usage (bool): Flag passed to the pipeline to reduce peak VRAM.
    """
    
    assert model_type in ['diff', 'determ']
    set_seed(seed)
    
    # --- MODEL LOADING ---
    print(f"Loading UNet ({model_type})...")
    unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
        'TencentARC/GeometryCrafter',
        subfolder='unet_diff' if model_type == 'diff' else 'unet_determ',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    ).requires_grad_(False).to("cuda", dtype=torch.float16)
    
    print("Loading Point Map VAE...")
    point_map_vae = PMapAutoencoderKLTemporalDecoder.from_pretrained(
        'TencentARC/GeometryCrafter',
        subfolder='point_map_vae',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        cache_dir=cache_dir
    ).requires_grad_(False).to("cuda", dtype=torch.float32)

    print("Loading Prior Model (MoGe)...")
    prior_model = MoGe(
        cache_dir=cache_dir,
    ).requires_grad_(False).to('cuda', dtype=torch.float32)

    # --- PIPELINE INITIALIZATION ---
    if model_type == 'diff':
        pipe = GeometryCrafterDiffPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=cache_dir
        ).to("cuda")
    else:
        pipe = GeometryCrafterDetermPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=cache_dir
        ).to("cuda")

    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(e)
        print("Xformers is not enabled")
    # bugs at https://github.com/continue-revolution/sd-webui-animatediff/issues/101
    # pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing()

    # --- VIDEO PROCESSING ---
    
    video_base_name = os.path.basename(video_path).split('.')[0]
    vid = VideoReader(video_path, ctx=cpu(0))
    original_height, original_width = vid.get_batch([0]).shape[1:3]

    if height is None or width is None:
        height = original_height
        width = original_width
    
    assert height % 64 == 0
    assert width % 64 == 0

    frames_idx = list(range(0, len(vid), process_stride))
    frames = vid.get_batch(frames_idx).asnumpy().astype(np.float32) / 255.0
    if process_length > 0:
        process_length = min(process_length, len(frames))
        frames = frames[:process_length]
    else:
        process_length = len(frames)
    window_size = min(window_size, process_length)
    if window_size == process_length: 
        overlap = 0
    frames_tensor = torch.tensor(frames.astype("float32"), device='cuda').float().permute(0, 3, 1, 2)
    # t,3,h,w

    if downsample_ratio > 1.0:
        original_height, original_width = frames_tensor.shape[-2], frames_tensor.shape[-1]
        frames_tensor = F.interpolate(frames_tensor, (round(frames_tensor.shape[-2]/downsample_ratio), round(frames_tensor.shape[-1]/downsample_ratio)), mode='bicubic', antialias=True).clamp(0, 1)

    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    # --- INFERENCE ---
    print(f"Starting Geometry Generation for {video_base_name}...")
    with torch.inference_mode():
        rec_point_map, rec_valid_mask = pipe(
            frames_tensor,
            point_map_vae,
            prior_model,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            window_size=window_size,
            decode_chunk_size=decode_chunk_size,
            overlap=overlap,
            force_projection=force_projection,
            force_fixed_focal=force_fixed_focal,
            use_extract_interp=use_extract_interp,
            track_time=track_time,
            low_memory_usage=low_memory_usage
        )

        if downsample_ratio > 1.0:
            rec_point_map = F.interpolate(rec_point_map.permute(0,3,1,2), (original_height, original_width), mode='bilinear').permute(0, 2, 3, 1)
            rec_valid_mask = F.interpolate(rec_valid_mask.float().unsqueeze(1), (original_height, original_width), mode='bilinear').squeeze(1) > 0.5

        np.savez(
            str(save_path / f"{video_base_name}.npz"), 
            point_map=rec_point_map.detach().cpu().numpy().astype(np.float16), 
            mask=rec_valid_mask.detach().cpu().numpy().astype(np.bool_))
        print(f"Done! Result saved to {out_file}")

if __name__ == "__main__":
    Fire(main)
