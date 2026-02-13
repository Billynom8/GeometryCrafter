"""Merger for assembling and blending aligned video segments.

This module provides functionality to merge multiple video segments into a
single sequence with seamless blending in overlapping regions.

Example:
    >>> from gc_seg import create_segment_mapping
    >>> from gc_seg.aligner import load_alignment
    >>> from gc_seg.merger import merge_segments
    >>> merge_segments(
    ...     segments_folder="workspace/temp_frames",
    ...     segment_mapping=mapping,
    ...     alignment_path="workspace/alignment.json",
    ...     output_folder="workspace/merged_frames",
    ... )
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from PIL import Image


class BlendMode(Enum):
    """Blending mode for overlapping regions."""

    LINEAR = "linear"
    SIGMOID = "sigmoid"


@dataclass
class MergerConfig:
    """Configuration for the segment merger.

    Attributes:
        segments_folder: Path to folder containing segment PNG directories.
        segment_mapping: SegmentMapping object from Phase 1.
        alignment_path: Path to alignment JSON from Phase 4.
        output_folder: Path to save merged PNG sequence.
        blend_mode: Blending mode for overlap regions.
        blend_sigma: Sigma parameter for sigmoid blending.
        bypass_alignment: If True, skip scale/shift alignment.
        clean_output: If True, clean output folder before merging.
    """

    segments_folder: str
    segment_mapping: Any
    alignment_path: str
    output_folder: str
    blend_mode: BlendMode = BlendMode.LINEAR
    blend_sigma: float = 6.0
    bypass_alignment: bool = False
    clean_output: bool = True


def load_png_as_float(path: Path) -> np.ndarray:
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
    arr = arr.astype(np.float32) / 65535.0
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def save_float_as_png(arr: np.ndarray, path: Path) -> None:
    """Saves a float array as 16-bit PNG.

    Args:
        arr: Float array with values in [0, 1].
        path: Path to save the PNG file.
    """
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    uint16 = (np.clip(arr, 0, 1) * 65535).astype(np.uint16)
    Image.fromarray(uint16).save(path, bits=16)


def apply_scale_shift(image: np.ndarray, scale: float, shift: float) -> np.ndarray:
    """Applies scale and shift transformation to an image.

    Args:
        image: Input image array with values in [0, 1].
        scale: Scale factor (s).
        shift: Shift factor (t).

    Returns:
        Transformed image with values in [0, 1].
    """
    transformed = image * scale + shift
    return np.clip(transformed, 0, 1)


def create_blend_weights(num_overlap: int, blend_mode: BlendMode, sigma: float = 6.0) -> np.ndarray:
    """Creates blending weights for overlap region.

    Args:
        num_overlap: Number of frames in the overlap.
        blend_mode: Blending mode to use.
        sigma: Sigma parameter for sigmoid (default: 6.0).

    Returns:
        Array of weights in [0, 1] for the second segment.
    """
    if blend_mode == BlendMode.LINEAR:
        t = np.arange(num_overlap, dtype=np.float32)
        weights = t / (num_overlap - 1) if num_overlap > 1 else np.zeros(num_overlap)
    else:
        t = np.linspace(-sigma, sigma, num_overlap, dtype=np.float32)
        weights = 1.0 / (1.0 + np.exp(-t))

    return weights


def blend_frames(frame_a: np.ndarray, frame_b: np.ndarray, weight: float) -> np.ndarray:
    """Blends two frames using a weight.

    Args:
        frame_a: Reference frame (first segment).
        frame_b: Aligned frame (second segment).
        weight: Blending weight for frame_b (0 = full frame_a, 1 = full frame_b).

    Returns:
        Blended frame.
    """
    return frame_a * (1 - weight) + frame_b * weight


class Merger:
    """Merger for assembling and blending video segments.

    This class handles reading segment PNGs, applying alignment transformations,
    and blending overlapping regions to produce a seamless output sequence.

    Attributes:
        config: Configuration for the merger.

    Example:
        >>> config = MergerConfig(
        ...     segments_folder="workspace/temp_frames",
        ...     segment_mapping=mapping,
        ...     alignment_path="workspace/alignment.json",
        ...     output_folder="workspace/merged_frames",
        ... )
        >>> merger = Merger(config)
        >>> merger.merge()
    """

    def __init__(self, config: MergerConfig):
        """Initializes the Merger.

        Args:
            config: Configuration for the merger.
        """
        self.config = config
        self.alignments = self._load_alignments()

    def _load_alignments(self) -> List[dict]:
        """Loads alignment coefficients from JSON.

        Returns:
            List of alignment dictionaries.
        """
        alignment_path = Path(self.config.alignment_path)
        if not alignment_path.exists():
            return []

        data = json.loads(alignment_path.read_text())
        return data.get("alignments", [])

    def _get_alignment_for_pair(self, segment_a: int, segment_b: int) -> Optional[dict]:
        """Gets alignment coefficients for a segment pair.

        Args:
            segment_a: Index of the first segment.
            segment_b: Index of the second segment.

        Returns:
            Alignment dictionary or None if not found.
        """
        for align in self.alignments:
            if align["segment_a"] == segment_a and align["segment_b"] == segment_b:
                return align
        return None

    def merge(self) -> List[Path]:
        """Merges all segments into a single sequence.

        Returns:
            List of paths to the merged PNG files.
        """
        segments_folder = Path(self.config.segments_folder)
        output_folder = Path(self.config.output_folder)

        # Clean output folder if requested
        if self.config.clean_output and output_folder.exists():
            import shutil

            for f in output_folder.glob("*.png"):
                f.unlink()

        output_folder.mkdir(parents=True, exist_ok=True)

        segments = self.config.segment_mapping.segments
        overlap = self.config.segment_mapping.overlap
        blend_weights = create_blend_weights(overlap, self.config.blend_mode, self.config.blend_sigma)

        if len(segments) == 1:
            return self._merge_single_segment(segments_folder, output_folder)

        output_paths_dict = {}  # Use dict to avoid duplicates from overlapping writes
        prev_segment_output_end = 0  # Where the next frame would go after previous segment

        for seg_idx in range(len(segments)):
            seg_dir = segments_folder / f"part_{seg_idx:04d}"
            if not seg_dir.exists():
                continue

            segment = segments[seg_idx]
            num_frames = segment.frame_count

            for frame_i in range(num_frames):
                frame_path = seg_dir / f"frame_{frame_i:04d}.png"
                if not frame_path.exists():
                    continue

                img = load_png_as_float(frame_path)

                if seg_idx > 0 and not self.config.bypass_alignment:
                    align = self._get_alignment_for_pair(seg_idx - 1, seg_idx)
                    if align:
                        img = apply_scale_shift(img, align["scale"], align["shift"])

                is_overlap_frame = seg_idx > 0 and frame_i < overlap

                if is_overlap_frame:
                    # Overlap frames replace the last N frames of previous segment
                    out_idx = prev_segment_output_end - overlap + frame_i
                else:
                    # Non-overlap frames continue from where previous segment ended
                    out_idx = prev_segment_output_end + (frame_i - overlap if seg_idx > 0 else frame_i)

                out_path = output_folder / f"frame_{out_idx:04d}.png"

                if is_overlap_frame and out_idx >= 0:
                    # Load the frame we're about to overwrite (from previous segment)
                    existing_path = output_folder / f"frame_{out_idx:04d}.png"
                    if existing_path.exists():
                        existing_img = load_png_as_float(existing_path)
                        weight = blend_weights[frame_i]
                        img = blend_frames(existing_img, img, weight)

                save_float_as_png(img, out_path)
                # Use dict to overwrite any previous entry for this index
                output_paths_dict[out_idx] = out_path

            # Update: after this segment, where would the next frame go?
            # It's the current end plus any new (non-overlap) frames from this segment
            if seg_idx == 0:
                prev_segment_output_end = num_frames
            else:
                prev_segment_output_end += num_frames - overlap

        # Sort by frame index and return paths
        output_paths = [output_paths_dict[i] for i in sorted(output_paths_dict.keys())]
        print(f"Merged {len(output_paths)} frames to {output_folder}")
        return output_paths

    def _merge_single_segment(self, segments_folder: Path, output_folder: Path) -> List[Path]:
        """Merges a single segment (no blending needed).

        Args:
            segments_folder: Path to segment directories.
            output_folder: Path to save merged frames.

        Returns:
            List of output paths.
        """
        seg_dir = segments_folder / "part_0000"
        if not seg_dir.exists():
            return []

        output_paths = []
        frame_files = sorted(seg_dir.glob("frame_*.png"))

        for i, frame_path in enumerate(frame_files):
            img = load_png_as_float(frame_path)
            out_path = output_folder / f"frame_{i:04d}.png"
            save_float_as_png(img, out_path)
            output_paths.append(out_path)

        print(f"Copied {len(output_paths)} frames to {output_folder}")
        return output_paths


def merge_segments(
    segments_folder: str,
    segment_mapping,
    alignment_path: str,
    output_folder: str,
    blend_mode: str = "linear",
    blend_sigma: float = 6.0,
    bypass_alignment: bool = False,
    clean_output: bool = True,
) -> List[Path]:
    """High-level function to merge aligned segments.

    This is the main entry point for Phase 5 of the GC-SEG pipeline.

    Args:
        segments_folder: Path to folder containing segment PNG directories.
        segment_mapping: SegmentMapping object from Phase 1.
        alignment_path: Path to alignment JSON from Phase 4.
        output_folder: Path to save merged PNG sequence.
        blend_mode: Blending mode - 'linear' or 'sigmoid' (default: 'linear').
        blend_sigma: Sigma parameter for sigmoid blending (default: 6.0).
        bypass_alignment: If True, skip scale/shift alignment.
        clean_output: If True, clean output folder before merging.

    Returns:
        List of paths to the merged PNG files.

    Example:
        >>> from gc_seg import create_segment_mapping
        >>> from gc_seg.merger import merge_segments
        >>> mapping = create_segment_mapping("video.mp4")
        >>> merge_segments(
        ...     segments_folder="workspace/temp_frames",
        ...     segment_mapping=mapping,
        ...     alignment_path="workspace/alignment.json",
        ...     output_folder="workspace/merged_frames",
        ... )
    """
    config = MergerConfig(
        segments_folder=segments_folder,
        segment_mapping=segment_mapping,
        alignment_path=alignment_path,
        output_folder=output_folder,
        blend_mode=BlendMode(blend_mode),
        blend_sigma=blend_sigma,
        bypass_alignment=bypass_alignment,
        clean_output=clean_output,
    )
    merger = Merger(config)
    return merger.merge()


if __name__ == "__main__":
    import argparse

    from gc_seg import create_segment_mapping

    parser = argparse.ArgumentParser(description="Merge aligned video segments")
    parser.add_argument(
        "--segments-folder", "-i", default="workspace/temp_frames", help="Folder containing segment PNG directories"
    )
    parser.add_argument("--video-path", "-v", required=True, help="Path to input video (for mapping)")
    parser.add_argument("--alignment-path", "-a", default="workspace/alignment.json", help="Path to alignment JSON")
    parser.add_argument(
        "--output-folder", "-o", default="workspace/merged_frames", help="Output folder for merged frames"
    )
    parser.add_argument(
        "--blend-mode", "-b", default="linear", choices=["linear", "sigmoid"], help="Blending mode for overlap regions"
    )
    parser.add_argument("--blend-sigma", type=float, default=6.0, help="Sigma parameter for sigmoid blending")
    parser.add_argument("--window-size", type=int, default=110, help="Window size used for segmentation")
    parser.add_argument("--overlap", type=int, default=25, help="Overlap between segments")

    args = parser.parse_args()

    mapping = create_segment_mapping(args.video_path, window_size=args.window_size, overlap=args.overlap)

    merge_segments(
        segments_folder=args.segments_folder,
        segment_mapping=mapping,
        alignment_path=args.alignment_path,
        output_folder=args.output_folder,
        blend_mode=args.blend_mode,
        blend_sigma=args.blend_sigma,
    )
