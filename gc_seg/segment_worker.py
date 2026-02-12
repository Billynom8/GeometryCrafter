"""Segment worker for running AI inference on video segments.

This module provides the SegmentWorker class for processing video segments
through the GeometryCrafter pipeline, producing NPZ files for each segment.

Example:
    >>> from gc_seg import create_segment_mapping, process_video_segments
    >>> mapping = create_segment_mapping("video.mp4", window_size=50, overlap=10)
    >>> paths = process_video_segments(
    ...     video_path="video.mp4",
    ...     segment_mapping=mapping,
    ...     save_folder="workspace/segments",
    ... )
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

from geometrycrafter import (
    GeometryCrafterDiffPipeline,
    GeometryCrafterDetermPipeline,
    PMapAutoencoderKLTemporalDecoder,
    UNetSpatioTemporalConditionModelVid2vid,
)
from third_party import MoGe

from gc_seg.metadata import SegmentMapping


@dataclass
class SegmentWorkerConfig:
    """Configuration for the SegmentWorker.

    Attributes:
        video_path: Path to the input video file.
        segment_mapping: Pre-computed segment mapping.
        save_folder: Directory to save NPZ segment files.
        cache_dir: Directory for model cache (default: workspace/cache).
        height: Output height in pixels (must be divisible by 64).
        width: Output width in pixels (must be divisible by 64).
        model_type: Model variant - 'diff' or 'determ'.
        num_inference_steps: Number of diffusion steps (default 5).
        guidance_scale: Classifier-free guidance scale (default 1.0).
        decode_chunk_size: Frames processed at once by VAE (default 8).
        force_projection: Enforce perspective projection constraints.
        force_fixed_focal: Assume constant camera focal length.
        downsample_ratio: Input downsample factor (default 1.0).
        seed: Random seed for reproducibility (default 42).
        low_memory_usage: Enable memory-efficient processing.
        track_time: Print timing information.
    """

    video_path: str
    segment_mapping: SegmentMapping
    save_folder: str
    cache_dir: str = "workspace/cache"
    height: Optional[int] = None
    width: Optional[int] = None
    model_type: str = "diff"
    num_inference_steps: int = 5
    guidance_scale: float = 1.0
    decode_chunk_size: int = 8
    force_projection: bool = True
    force_fixed_focal: bool = True
    downsample_ratio: float = 1.0
    seed: int = 42
    low_memory_usage: bool = False
    track_time: bool = False


class SegmentWorker:
    """Worker class for processing video segments through GeometryCrafter.

    This class handles loading models, processing each segment, and saving
    the results as float16 NPZ files.

    Attributes:
        config: Configuration for the worker.

    Example:
        >>> config = SegmentWorkerConfig(
        ...     video_path="video.mp4",
        ...     segment_mapping=mapping,
        ...     save_folder="workspace/segments",
        ... )
        >>> worker = SegmentWorker(config)
        >>> paths = worker.process_all_segments()
        >>> worker.cleanup()
    """

    def __init__(self, config: SegmentWorkerConfig):
        """Initializes the SegmentWorker.

        Args:
            config: Configuration for the worker.
        """
        self.config = config
        self.pipe = None
        self.point_map_vae = None
        self.prior_model = None

    def _load_models(self):
        """Loads all required models for inference.

        Loads UNet, Point Map VAE, Prior Model (MoGe), and the pipeline.
        All models are moved to CUDA and set to inference mode.
        """
        cfg = self.config
        from diffusers.training_utils import set_seed

        set_seed(cfg.seed)

        print(f"Loading UNet ({cfg.model_type})...")
        unet = (
            UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
                "TencentARC/GeometryCrafter",
                subfolder="unet_diff" if cfg.model_type == "diff" else "unet_determ",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                cache_dir=cfg.cache_dir,
            )
            .requires_grad_(False)
            .to("cuda", dtype=torch.float16)
        )

        print("Loading Point Map VAE...")
        self.point_map_vae = (
            PMapAutoencoderKLTemporalDecoder.from_pretrained(
                "TencentARC/GeometryCrafter",
                subfolder="point_map_vae",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                cache_dir=cfg.cache_dir,
            )
            .requires_grad_(False)
            .to("cuda", dtype=torch.float32)
        )

        print("Loading Prior Model (MoGe)...")
        self.prior_model = MoGe(cache_dir=cfg.cache_dir).requires_grad_(False).to("cuda", dtype=torch.float32)

        print(f"Loading Pipeline ({cfg.model_type})...")
        if cfg.model_type == "diff":
            self.pipe = GeometryCrafterDiffPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                unet=unet,
                torch_dtype=torch.float16,
                variant="fp16",
                cache_dir=cfg.cache_dir,
            ).to("cuda")
        else:
            self.pipe = GeometryCrafterDetermPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                unet=unet,
                torch_dtype=torch.float16,
                variant="fp16",
                cache_dir=cfg.cache_dir,
            ).to("cuda")

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Xformers not enabled: {e}")
        self.pipe.enable_attention_slicing()

    def _prepare_video_frames(self, start_frame: int, end_frame: int) -> torch.Tensor:
        """Prepares video frames for inference.

        Loads frames from the video file and applies any required preprocessing
        such as downsampling.

        Args:
            start_frame: Zero-indexed start frame.
            end_frame: Zero-indexed end frame (inclusive).

        Returns:
            Tensor of shape [T, C, H, W] in range [0, 1].

        Raises:
            AssertionError: If height or width is not divisible by 64.
        """
        cfg = self.config
        vid = VideoReader(cfg.video_path, ctx=cpu(0))

        if cfg.height is None or cfg.width is None:
            cfg.height = vid[0].shape[0]
            cfg.width = vid[0].shape[1]

        assert cfg.height % 64 == 0
        assert cfg.width % 64 == 0

        frames_idx = list(range(start_frame, end_frame + 1))
        frames = vid.get_batch(frames_idx).asnumpy().astype(np.float32) / 255.0
        frames_tensor = torch.tensor(frames, device="cuda").float().permute(0, 3, 1, 2)

        if cfg.downsample_ratio > 1.0:
            frames_tensor = F.interpolate(
                frames_tensor,
                (
                    round(frames_tensor.shape[-2] / cfg.downsample_ratio),
                    round(frames_tensor.shape[-1] / cfg.downsample_ratio),
                ),
                mode="bicubic",
                antialias=True,
            ).clamp(0, 1)

        return frames_tensor

    def process_segment(self, segment_index: int) -> Path:
        """Processes a single video segment.

        Args:
            segment_index: Index of the segment to process.

        Returns:
            Path to the saved NPZ file.

        Raises:
            ValueError: If segment_index is out of range.
        """
        cfg = self.config
        mapping = cfg.segment_mapping

        if segment_index >= len(mapping.segments):
            raise ValueError(f"Segment index {segment_index} out of range (0-{len(mapping.segments) - 1})")

        segment = mapping.segments[segment_index]

        print(f"Processing {segment.name} (frames {segment.start_frame} to {segment.end_frame})...")

        frames_tensor = self._prepare_video_frames(segment.start_frame, segment.end_frame)

        window_size = segment.frame_count

        with torch.inference_mode():
            point_map, valid_mask = self.pipe(
                frames_tensor,
                self.point_map_vae,
                self.prior_model,
                height=cfg.height,
                width=cfg.width,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                window_size=window_size,
                decode_chunk_size=cfg.decode_chunk_size,
                overlap=0,
                force_projection=cfg.force_projection,
                force_fixed_focal=cfg.force_fixed_focal,
                track_time=cfg.track_time,
                low_memory_usage=cfg.low_memory_usage,
            )

        if cfg.downsample_ratio > 1.0:
            orig_h = frames_tensor.shape[-2] * cfg.downsample_ratio
            orig_w = frames_tensor.shape[-1] * cfg.downsample_ratio
            point_map = F.interpolate(
                point_map.permute(0, 3, 1, 2), (int(orig_h), int(orig_w)), mode="bilinear"
            ).permute(0, 2, 3, 1)
            valid_mask = (
                F.interpolate(valid_mask.float().unsqueeze(1), (int(orig_h), int(orig_w)), mode="bilinear").squeeze(1)
                > 0.5
            )

        save_path = Path(cfg.save_folder)
        save_path.mkdir(parents=True, exist_ok=True)

        segment_filename = f"part_{segment_index:04d}.npz"
        output_path = save_path / segment_filename

        np.savez(
            str(output_path),
            point_map=point_map.detach().cpu().numpy().astype(np.float16),
            mask=valid_mask.detach().cpu().numpy().astype(np.bool_),
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            frame_count=segment.frame_count,
        )

        print(f"Saved {segment.name} to {output_path}")
        return output_path

    def process_all_segments(self) -> List[Path]:
        """Processes all segments in the segment mapping.

        Returns:
            List of paths to the saved NPZ files.
        """
        if self.pipe is None:
            self._load_models()

        mapping = self.config.segment_mapping
        output_paths = []

        for i in range(len(mapping.segments)):
            path = self.process_segment(i)
            output_paths.append(path)

        print(f"Processed {len(output_paths)} segments")
        return output_paths

    def cleanup(self):
        """Releases GPU memory by deleting models and clearing cache."""
        if self.prior_model is not None:
            del self.prior_model
        if self.point_map_vae is not None:
            del self.point_map_vae
        if self.pipe is not None:
            del self.pipe
        torch.cuda.empty_cache()
        import gc

        gc.collect()


def process_video_segments(
    video_path: str,
    segment_mapping: SegmentMapping,
    save_folder: str,
    cache_dir: str = "workspace/cache",
    height: Optional[int] = None,
    width: Optional[int] = None,
    model_type: str = "diff",
    num_inference_steps: int = 5,
    guidance_scale: float = 1.0,
    decode_chunk_size: int = 8,
    seed: int = 42,
    low_memory_usage: bool = False,
) -> List[Path]:
    """High-level function to process video segments.

    This is the main entry point for Phase 2 of the GC-SEG pipeline.

    Args:
        video_path: Path to the input video file.
        segment_mapping: Pre-computed segment mapping.
        save_folder: Directory to save NPZ segment files.
        cache_dir: Directory for model cache.
        height: Output height (must be divisible by 64).
        width: Output width (must be divisible by 64).
        model_type: 'diff' or 'determ'.
        num_inference_steps: Number of diffusion steps.
        guidance_scale: Classifier-free guidance scale.
        decode_chunk_size: Frames processed at once by VAE.
        seed: Random seed for reproducibility.
        low_memory_usage: Enable memory-efficient processing.

    Returns:
        List of paths to the saved NPZ files.

    Example:
        >>> from gc_seg import create_segment_mapping, process_video_segments
        >>> mapping = create_segment_mapping("video.mp4", window_size=50)
        >>> paths = process_video_segments(
        ...     video_path="video.mp4",
        ...     segment_mapping=mapping,
        ...     save_folder="workspace/segments",
        ...     height=576,
        ...     width=1024,
        ... )
    """
    config = SegmentWorkerConfig(
        video_path=video_path,
        segment_mapping=segment_mapping,
        save_folder=save_folder,
        cache_dir=cache_dir,
        height=height,
        width=width,
        model_type=model_type,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        decode_chunk_size=decode_chunk_size,
        seed=seed,
        low_memory_usage=low_memory_usage,
    )

    worker = SegmentWorker(config)
    try:
        return worker.process_all_segments()
    finally:
        worker.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segment Worker for GeometryCrafter")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--mapping", "-m", required=True, help="Path to segment mapping JSON")
    parser.add_argument("--save-folder", "-o", default="workspace/segments", help="Output folder")
    parser.add_argument("--height", type=int, help="Output height (multiple of 64)")
    parser.add_argument("--width", type=int, help="Output width (multiple of 64)")
    parser.add_argument("--model-type", default="diff", choices=["diff", "determ"])
    parser.add_argument("--num-inference-steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--decode-chunk-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-memory-usage", action="store_true")
    parser.add_argument("--track-time", action="store_true")

    args = parser.parse_args()

    mapping = SegmentMapping.from_json(Path(args.mapping), args.video_path)

    config = SegmentWorkerConfig(
        video_path=args.video_path,
        segment_mapping=mapping,
        save_folder=args.save_folder,
        height=args.height,
        width=args.width,
        model_type=args.model_type,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        decode_chunk_size=args.decode_chunk_size,
        seed=args.seed,
        low_memory_usage=args.low_memory_usage,
    )

    worker = SegmentWorker(config)
    try:
        worker.process_all_segments()
    finally:
        worker.cleanup()
