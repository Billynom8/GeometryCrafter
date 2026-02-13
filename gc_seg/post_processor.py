"""Post-processor for applying bilateral filter and encoding to video.

This module provides functionality to apply joint bilateral filtering to merged
disparity frames and encode the result as a high-quality x265 10-bit video.

Example:
    >>> from gc_seg import create_segment_mapping
    >>> from gc_seg.post_processor import process_video
    >>> process_video(
    ...     merged_frames_folder="workspace/merged_frames",
    ...     original_video="video.mp4",
    ...     output_path="workspace/output.mp4",
    ... )
"""

import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


class Encoder(Enum):
    """Video encoder to use."""

    X265 = "x265"
    VP9 = "vp9"
    H264 = "h264"
    NVENC_H264 = "nvenc_h264"
    NVENC_HEVC = "nvenc_hevc"


@dataclass
class PostProcessorConfig:
    """Configuration for post-processing.

    Attributes:
        merged_frames_folder: Path to merged PNG frames.
        original_video: Path to original video for joint filtering.
        output_path: Path for output MP4 file.
        encoder: Video encoder to use.
        encoder_preset: Encoding speed preset.
        encoder_crf: Constant Rate Factor (lower = higher quality).
        bilateral_d: Diameter of pixel neighborhood.
        bilateral_sigma_color: Filter sigma in color space.
        bilateral_sigma_space: Filter sigma in coordinate space.
        fps: Frames per second for output video.
        bitdepth: Output bit depth (8 or 10).
        bypass_bilateral: If True, skip bilateral upscaling and save at depthmap resolution.
        upscale_resolution: Tuple of (width, height) to upscale before encoding. None to keep current resolution.
    """

    merged_frames_folder: str
    original_video: str
    output_path: str
    encoder: Encoder = Encoder.X265
    encoder_preset: str = "medium"
    encoder_crf: int = 18
    bilateral_d: int = 9
    bilateral_sigma_color: float = 0.1
    bilateral_sigma_space: float = 0.1
    fps: float = 30.0
    bitdepth: int = 10
    bypass_bilateral: bool = False
    upscale_resolution: Optional[Tuple[int, int]] = None


def load_16bit_png(path: Path) -> np.ndarray:
    """Loads a 16-bit PNG as float array.

    Args:
        path: Path to the PNG file.

    Returns:
        Float array with values in [0, 1].
    """
    img = Image.open(path)
    if img.mode == "I;16":
        arr = np.array(img, dtype=np.uint16)
    else:
        arr = np.array(img, dtype=np.uint8)
    return arr.astype(np.float32) / 65535.0


def apply_joint_bilateral_filter(
    disparity: np.ndarray, reference: np.ndarray, d: int = 9, sigma_color: float = 0.1, sigma_space: float = 0.1
) -> np.ndarray:
    """Applies joint bilateral filter to disparity map.

    Uses the reference image to guide the filtering, preserving edges.

    Args:
        disparity: Disparity/depth map (0-1 range).
        reference: Reference RGB image (0-1 range).
        d: Diameter of pixel neighborhood.
        sigma_color: Filter sigma in color space.
        sigma_space: Filter sigma in coordinate space.

    Returns:
        Filtered disparity map.
    """
    disp_u16 = (disparity * 65535).astype(np.uint16)
    ref_u8 = (reference * 255).astype(np.uint8)

    if hasattr(cv2, "ximgproc_JointBilateralFilter"):
        filtered = cv2.ximgproc_JointBilateralFilter(
            ref_u8, disp_u16, d=d, sigmaColor=sigma_color * 255, sigmaSpace=sigma_space * 255
        )
    else:
        disp_f32 = (disparity * 65535).astype(np.float32)
        filtered = cv2.bilateralFilter(disp_f32, d, sigma_color * 255, sigma_space * 255)
        filtered = filtered.astype(np.float32) / 65535.0

    return filtered


def frames_to_video_ffmpeg(
    frames_folder: Path,
    output_path: Path,
    fps: float = 30.0,
    bitdepth: int = 10,
    encoder: Encoder = Encoder.X265,
    preset: str = "medium",
    crf: int = 18,
) -> None:
    """Encodes frames to video using FFmpeg.

    Args:
        frames_folder: Path to folder containing PNG frames.
        output_path: Path for output video file.
        fps: Frames per second.
        bitdepth: Output bit depth (8 or 10).
        encoder: Video encoder to use.
        preset: Encoding preset (ultrafast to veryslow for CPU, p1-p7 for NVENC).
        crf: Constant Rate Factor (0-51, lower = better quality). For NVENC, use cq (0-51).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pattern = frames_folder / "frame_%04d.png"
    first_frame = sorted(frames_folder.glob("frame_*.png"))[0]

    # Build command based on encoder
    if encoder == Encoder.NVENC_H264:
        codec = "h264_nvenc"
        pix_fmt = "yuv420p"
        # NVENC uses -cq instead of -crf, and -preset p1-p7
        nvenc_preset = preset if preset.startswith("p") else "p4"
        quality_args = ["-cq", str(crf), "-preset", nvenc_preset]
        extra_args = ["-pix_fmt", pix_fmt] + quality_args
    elif encoder == Encoder.NVENC_HEVC:
        codec = "hevc_nvenc"
        pix_fmt = "yuv420p10le" if bitdepth == 10 else "yuv420p"
        nvenc_preset = preset if preset.startswith("p") else "p4"
        quality_args = ["-cq", str(crf), "-preset", nvenc_preset]
        extra_args = ["-pix_fmt", pix_fmt] + quality_args
    elif bitdepth == 10 and encoder == Encoder.X265:
        pix_fmt = "yuv420p10le"
        codec = "libx265"
        extra_args = ["-pix_fmt", pix_fmt, "-crf", str(crf), "-x265-params", "profile=main10"]
    elif encoder == Encoder.X265:
        pix_fmt = "yuv420p"
        codec = "libx265"
        extra_args = ["-pix_fmt", pix_fmt, "-crf", str(crf)]
    elif encoder == Encoder.VP9:
        pix_fmt = "yuv420p"
        codec = "libvpx-vp9"
        extra_args = ["-pix_fmt", pix_fmt, "-crf", str(crf), "-b:v", "0"]
    else:  # H264
        pix_fmt = "yuv420p"
        codec = "libx264"
        extra_args = ["-pix_fmt", pix_fmt, "-crf", str(crf)]

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(pattern),
        "-c:v",
        codec,
        *extra_args,
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    if encoder == Encoder.X265:
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(pattern),
            "-c:v",
            "libx265",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p10le" if bitdepth == 10 else "yuv420p",
            "-x265-params",
            "profile=main10" if bitdepth == 10 else "",
            str(output_path),
        ]
        cmd = [c for c in cmd if c]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    print(f"Encoded video to {output_path}")


class PostProcessor:
    """Post-processor for applying bilateral filter and encoding.

    This class handles applying joint bilateral filtering to merged disparity
    frames and encoding the result as a high-quality video.

    Attributes:
        config: Configuration for post-processing.

    Example:
        >>> config = PostProcessorConfig(
        ...     merged_frames_folder="workspace/merged_frames",
        ...     original_video="video.mp4",
        ...     output_path="workspace/output.mp4",
        ... )
        >>> processor = PostProcessor(config)
        >>> processor.process()
    """

    def __init__(self, config: PostProcessorConfig):
        """Initializes the PostProcessor.

        Args:
            config: Configuration for post-processing.
        """
        self.config = config

    def _load_reference_frames(self) -> List[np.ndarray]:
        """Loads frames from the original video.

        Returns:
            List of RGB frames as numpy arrays.
        """
        import decord
        from decord import cpu

        vr = decord.VideoReader(self.config.original_video, ctx=cpu(0))
        frames = vr.get_batch(range(len(vr))).asnumpy()
        return [frame.astype(np.float32) / 255.0 for frame in frames]

    def _get_output_fps(self) -> float:
        """Gets FPS from original video metadata.

        Returns:
            FPS value.
        """
        import decord
        from decord import cpu

        vr = decord.VideoReader(self.config.original_video, ctx=cpu(0))
        fps = vr.get_avg_fps()
        return fps if fps > 0 else self.config.fps

    def process(self) -> Path:
        """Applies bilateral filter and encodes the video.

        Returns:
            Path to the output video file.
        """
        cfg = self.config
        frames_folder = Path(cfg.merged_frames_folder)
        output_path = Path(cfg.output_path)

        frame_files = sorted(frames_folder.glob("frame_*.png"))
        if not frame_files:
            raise ValueError(f"No frames found in {frames_folder}")

        fps = self._get_output_fps()

        filtered_folder = frames_folder.parent / "filtered_frames"
        # Clean filtered folder if it exists (remove old frames from previous runs)
        if filtered_folder.exists():
            import shutil

            for f in filtered_folder.glob("*.png"):
                f.unlink()
        filtered_folder.mkdir(parents=True, exist_ok=True)

        if cfg.bypass_bilateral:
            # Skip bilateral upscaling, just copy frames at their current resolution
            print(f"Bypassing bilateral filter, copying {len(frame_files)} frames...")
            import shutil

            for frame_path in frame_files:
                shutil.copy(frame_path, filtered_folder / frame_path.name)
            print(f"Copied frames to {filtered_folder}")
        else:
            # Apply joint bilateral filter to upscale
            ref_frames = self._load_reference_frames()
            print(f"Applying joint bilateral filter to {len(frame_files)} frames...")

            for i, frame_path in enumerate(frame_files):
                disparity = load_16bit_png(frame_path)

                if i < len(ref_frames):
                    ref = ref_frames[i]
                else:
                    ref = ref_frames[-1]

                if cfg.bilateral_d > 0:
                    filtered = apply_joint_bilateral_filter(
                        disparity,
                        ref,
                        d=cfg.bilateral_d,
                        sigma_color=cfg.bilateral_sigma_color,
                        sigma_space=cfg.bilateral_sigma_space,
                    )
                else:
                    filtered = disparity

                out_path = filtered_folder / frame_path.name
                uint16 = (np.clip(filtered, 0, 1) * 65535).astype(np.uint16)
                Image.fromarray(uint16).save(out_path, bits=16)

                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(frame_files)} frames")

            print(f"Saving filtered frames to {filtered_folder}")

        # Optional: Upscale to target resolution before encoding
        if cfg.upscale_resolution is not None:
            target_w, target_h = cfg.upscale_resolution
            print(f"Upscaling frames to {target_w}x{target_h}...")
            import cv2

            upscaled_folder = filtered_folder.parent / "upscaled_frames"
            if upscaled_folder.exists():
                for f in upscaled_folder.glob("*.png"):
                    f.unlink()
            upscaled_folder.mkdir(parents=True, exist_ok=True)

            frame_files = sorted(filtered_folder.glob("frame_*.png"))
            for frame_path in frame_files:
                img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
                upscaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(str(upscaled_folder / frame_path.name), upscaled)

            print(f"Upscaled {len(frame_files)} frames")
            filtered_folder = upscaled_folder

        frames_to_video_ffmpeg(
            frames_folder=filtered_folder,
            output_path=output_path,
            fps=fps,
            bitdepth=cfg.bitdepth,
            encoder=cfg.encoder,
            preset=cfg.encoder_preset,
            crf=cfg.encoder_crf,
        )

        return output_path


def process_video(
    merged_frames_folder: str,
    original_video: str,
    output_path: str,
    encoder: str = "x265",
    encoder_preset: str = "medium",
    encoder_crf: int = 18,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 0.1,
    bilateral_sigma_space: float = 0.1,
    fps: Optional[float] = None,
    bitdepth: int = 10,
    bypass_bilateral: bool = False,
    upscale_resolution: Optional[Tuple[int, int]] = None,
) -> Path:
    """High-level function to post-process merged frames.

    This is the main entry point for Phase 6 of the GC-SEG pipeline.

    Args:
        merged_frames_folder: Path to merged PNG frames.
        original_video: Path to original video for joint filtering.
        output_path: Path for output MP4 file.
        encoder: Video encoder - 'x265', 'vp9', or 'h264'.
        encoder_preset: Encoding speed preset.
        encoder_crf: Constant Rate Factor.
        bilateral_d: Diameter for bilateral filter (0 to disable).
        bilateral_sigma_color: Sigma color for bilateral filter.
        bilateral_sigma_space: Sigma space for bilateral filter.
        fps: Output FPS (auto-detect from original if None).
        bitdepth: Output bit depth (8 or 10).
        bypass_bilateral: If True, skip bilateral upscaling and save at depthmap resolution.
        upscale_resolution: Tuple of (width, height) to upscale before encoding. None to keep current resolution.

    Returns:
        Path to the output video file.

    Example:
        >>> from gc_seg.post_processor import process_video
        >>> process_video(
        ...     merged_frames_folder="workspace/merged_frames",
        ...     original_video="video.mp4",
        ...     output_path="workspace/output.mp4",
        ...     bilateral_d=9,
        ...     bitdepth=10,
        ... )
    """
    # Handle both string and enum encoder values
    if isinstance(encoder, Encoder):
        encoder_enum = encoder
    elif hasattr(encoder, "value"):
        # It's an enum from a potentially different import, get its value
        encoder_enum = Encoder(encoder.value)
    else:
        encoder_enum = Encoder(str(encoder))

    config = PostProcessorConfig(
        merged_frames_folder=merged_frames_folder,
        original_video=original_video,
        output_path=output_path,
        encoder=encoder_enum,
        encoder_preset=encoder_preset,
        encoder_crf=encoder_crf,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
        fps=fps or 30.0,
        bitdepth=bitdepth,
        bypass_bilateral=bypass_bilateral,
        upscale_resolution=upscale_resolution,
    )
    processor = PostProcessor(config)
    return processor.process()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-process merged frames to video")
    parser.add_argument(
        "--merged-frames-folder", "-i", default="workspace/merged_frames", help="Folder containing merged PNG frames"
    )
    parser.add_argument("--original-video", "-v", required=True, help="Path to original video")
    parser.add_argument("--output-path", "-o", default="workspace/output.mp4", help="Output video path")
    parser.add_argument("--encoder", "-e", default="x265", choices=["x265", "vp9", "h264"], help="Video encoder")
    parser.add_argument("--preset", default="medium", help="Encoding preset (ultrafast to veryslow)")
    parser.add_argument("--crf", type=int, default=18, help="Constant Rate Factor (lower = better quality)")
    parser.add_argument("--bilateral-d", type=int, default=9, help="Bilateral filter diameter (0 to disable)")
    parser.add_argument("--bilateral-sigma-color", type=float, default=0.1, help="Bilateral filter sigma color")
    parser.add_argument("--bilateral-sigma-space", type=float, default=0.1, help="Bilateral filter sigma space")
    parser.add_argument("--fps", type=float, default=None, help="Output FPS (auto-detect from original if not set)")
    parser.add_argument("--bitdepth", type=int, default=10, choices=[8, 10], help="Output bit depth")

    args = parser.parse_args()

    process_video(
        merged_frames_folder=args.merged_frames_folder,
        original_video=args.original_video,
        output_path=args.output_path,
        encoder=args.encoder,
        encoder_preset=args.preset,
        encoder_crf=args.crf,
        bilateral_d=args.bilateral_d,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
        fps=args.fps,
        bitdepth=args.bitdepth,
    )
