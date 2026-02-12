"""Aligner for computing scale and shift alignment between video segments.

This module provides functionality to calculate the scale (s) and shift (t)
coefficients needed to align overlapping frames between video segments
using least squares optimization.

Example:
    >>> from gc_seg import create_segment_mapping
    >>> from gc_seg.aligner import compute_alignment
    >>> alignments = compute_alignment(
    ...     segments_folder="workspace/temp_frames",
    ...     segment_mapping=mapping,
    ...     output_path="workspace/alignment.json",
    ... )
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class AlignmentCoefficients:
    """Alignment coefficients for a segment pair.

    Attributes:
        segment_a_index: Index of the reference segment.
        segment_b_index: Index of the segment to align.
        scale: Scale factor (s) in the formula: new = s * old + t.
        shift: Shift factor (t) in the formula: new = s * old + t.
        overlap_frames: Number of overlapping frames used for calculation.
        rmse: Root mean square error after alignment.
    """

    segment_a_index: int
    segment_b_index: int
    scale: float
    shift: float
    overlap_frames: int
    rmse: float


def load_png_as_float(path: Path) -> np.ndarray:
    """Loads a 16-bit PNG as float array.

    Args:
        path: Path to the PNG file.

    Returns:
        Float array with values normalized to [0, 1].
    """
    img = Image.open(path)
    if img.mode == "I;16":
        arr = np.array(img, dtype=np.uint16)
    else:
        arr = np.array(img, dtype=np.uint8)
    return arr.astype(np.float32) / 65535.0


def compute_scale_shift(reference: np.ndarray, to_align: np.ndarray) -> Tuple[float, float, float]:
    """Computes scale and shift using least squares.

    Finds s and t such that: reference ≈ s * to_align + t

    Args:
        reference: Reference image array.
        to_align: Image to align to reference.

    Returns:
        Tuple of (scale, shift, rmse).

    Raises:
        ValueError: If arrays have different shapes or are empty.
    """
    if reference.shape != to_align.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {to_align.shape}")

    x = to_align.flatten().astype(np.float64)
    y = reference.flatten().astype(np.float64)

    valid_mask = (x > 0) & (y > 0)
    if not valid_mask.any():
        raise ValueError("No valid pixels in overlapping region")

    x = x[valid_mask]
    y = y[valid_mask]

    n = len(x)
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_xx = (x * x).sum()

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-10:
        scale = 1.0
        shift = 0.0
    else:
        scale = (n * sum_xy - sum_x * sum_y) / denom
        shift = (sum_y - scale * sum_x) / n

    aligned = scale * x + shift
    rmse = np.sqrt(((y - aligned) ** 2).mean())

    return float(scale), float(shift), float(rmse)


def align_segment_pair(
    segment_a_dir: Path, segment_b_dir: Path, overlap_start_a: int, overlap_start_b: int, num_overlap_frames: int
) -> AlignmentCoefficients:
    """Aligns a pair of segments using their overlapping frames.

    Args:
        segment_a_dir: Path to segment A PNG directory.
        segment_b_dir: Path to segment B PNG directory.
        overlap_start_a: Starting frame index in segment A.
        overlap_start_b: Starting frame index in segment B.
        num_overlap_frames: Number of overlapping frames.

    Returns:
        AlignmentCoefficients with scale, shift, and error metrics.
    """
    scales = []
    shifts = []
    rmses = []

    for i in range(num_overlap_frames):
        frame_a = segment_a_dir / f"frame_{overlap_start_a + i:04d}.png"
        frame_b = segment_b_dir / f"frame_{overlap_start_b + i:04d}.png"

        if not frame_a.exists() or not frame_b.exists():
            continue

        img_a = load_png_as_float(frame_a)
        img_b = load_png_as_float(frame_b)

        try:
            s, t, rmse = compute_scale_shift(img_a, img_b)
            scales.append(s)
            shifts.append(t)
            rmses.append(rmse)
        except ValueError:
            continue

    if not scales:
        raise ValueError("No valid overlapping frames found")

    median_scale = float(np.median(scales))
    median_shift = float(np.median(shifts))
    median_rmse = float(np.median(rmses))

    return AlignmentCoefficients(
        segment_a_index=0,
        segment_b_index=0,
        scale=median_scale,
        shift=median_shift,
        overlap_frames=len(scales),
        rmse=median_rmse,
    )


def compute_alignment(
    segments_folder: str, segment_mapping, output_path: Optional[Path] = None
) -> List[AlignmentCoefficients]:
    """Computes alignment coefficients between all adjacent segment pairs.

    This is the main entry point for Phase 4 of the GC-SEG pipeline.

    Args:
        segments_folder: Path to the folder containing segment PNG directories.
        segment_mapping: SegmentMapping object from Phase 1.
        output_path: Optional path to save alignment JSON.

    Returns:
        List of AlignmentCoefficients for each segment pair.

    Example:
        >>> from gc_seg import create_segment_mapping
        >>> from gc_seg.aligner import compute_alignment
        >>> mapping = create_segment_mapping("video.mp4")
        >>> alignments = compute_alignment(
        ...     segments_folder="workspace/temp_frames",
        ...     segment_mapping=mapping,
        ...     output_path="workspace/alignment.json",
        ... )
    """
    segments_folder = Path(segments_folder)
    segments = segment_mapping.segments

    if len(segments) < 2:
        return []

    alignments = []
    overlap = segment_mapping.overlap

    for i in range(len(segments) - 1):
        seg_a = segments[i]
        seg_b = segments[i + 1]

        dir_a = segments_folder / f"part_{i:04d}"
        dir_b = segments_folder / f"part_{i + 1:04d}"

        if not dir_a.exists() or not dir_b.exists():
            continue

        overlap_start_a = seg_a.frame_count - overlap
        overlap_start_b = 0

        coeffs = align_segment_pair(dir_a, dir_b, overlap_start_a, overlap_start_b, overlap)
        coeffs.segment_a_index = i
        coeffs.segment_b_index = i + 1
        alignments.append(coeffs)

        print(f"Aligned segment {i} -> {i + 1}: s={coeffs.scale:.6f}, t={coeffs.shift:.6f}, rmse={coeffs.rmse:.6f}")

    if output_path and alignments:
        output_data = {
            "overlap": overlap,
            "alignments": [
                {
                    "segment_a": a.segment_a_index,
                    "segment_b": a.segment_b_index,
                    "scale": a.scale,
                    "shift": a.shift,
                    "overlap_frames": a.overlap_frames,
                    "rmse": a.rmse,
                }
                for a in alignments
            ],
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"Saved alignment to {output_path}")

    return alignments


def load_alignment(path: Path) -> List[AlignmentCoefficients]:
    """Loads alignment coefficients from a JSON file.

    Args:
        path: Path to the alignment JSON file.

    Returns:
        List of AlignmentCoefficients.
    """
    data = json.loads(Path(path).read_text())
    alignments = []
    for a in data["alignments"]:
        alignments.append(
            AlignmentCoefficients(
                segment_a_index=a["segment_a"],
                segment_b_index=a["segment_b"],
                scale=a["scale"],
                shift=a["shift"],
                overlap_frames=a["overlap_frames"],
                rmse=a["rmse"],
            )
        )
    return alignments


if __name__ == "__main__":
    import argparse

    from gc_seg import create_segment_mapping

    parser = argparse.ArgumentParser(description="Compute alignment between segments")
    parser.add_argument(
        "--segments-folder", "-i", default="workspace/temp_frames", help="Folder containing segment PNG directories"
    )
    parser.add_argument("--video-path", "-v", required=True, help="Path to input video (for mapping)")
    parser.add_argument("--output", "-o", default="workspace/alignment.json", help="Output JSON path")
    parser.add_argument("--window-size", type=int, default=110, help="Window size used for segmentation")
    parser.add_argument("--overlap", type=int, default=25, help="Overlap between segments")

    args = parser.parse_args()

    mapping = create_segment_mapping(args.video_path, window_size=args.window_size, overlap=args.overlap)

    compute_alignment(segments_folder=args.segments_folder, segment_mapping=mapping, output_path=Path(args.output))
