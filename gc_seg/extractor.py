"""Extractor for converting NPZ segments to disparity PNG frames.

This module provides functionality to extract the Z-channel (depth/disparity)
from point map NPZ files and convert them to 16-bit PNG images.

Example:
    >>> from gc_seg import extract_disparity_frames
    >>> paths = extract_disparity_frames(
    ...     segments_folder="workspace/segments",
    ...     output_folder="workspace/temp_frames",
    ... )
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class ExtractorConfig:
    """Configuration for the disparity extractor.

    Attributes:
        segments_folder: Directory containing NPZ segment files.
        output_folder: Directory to save PNG frames.
        file_prefix: Prefix for NPZ files (default: 'part_').
        file_ext: Extension for NPZ files (default: '.npz').
        normalize: Whether to normalize depth values (default: True).
        invert: Whether to invert depth values (default: False).
    """

    segments_folder: str
    output_folder: str
    file_prefix: str = "part_"
    file_ext: str = ".npz"
    normalize: bool = True
    invert: bool = False


def load_npz_segment(path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Loads a segment NPZ file.

    Args:
        path: Path to the NPZ file.

    Returns:
        Tuple of (point_map, mask, metadata).
        - point_map: Array of shape [T, H, W, 3] containing X, Y, Z channels.
        - mask: Boolean array of shape [T, H, W] indicating valid pixels.
        - metadata: Dictionary with 'start_frame', 'end_frame', 'frame_count'.
    """
    data = np.load(path)
    point_map = data["point_map"]
    mask = data["mask"]
    metadata = {
        "start_frame": int(data.get("start_frame", 0)),
        "end_frame": int(data.get("end_frame", point_map.shape[0] - 1)),
        "frame_count": int(data.get("frame_count", point_map.shape[0])),
    }
    return point_map, mask, metadata


def extract_depth_from_point_map(point_map: np.ndarray, mask: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Extracts depth from a point map.

    Extracts the Z-channel from a point map and optionally normalizes it
    to the [0, 1] range using percentile-based scaling.

    Args:
        point_map: Array of shape [H, W, 3] with X, Y, Z channels.
        mask: Boolean array of shape [H, W] indicating valid pixels.
        normalize: Whether to normalize to [0, 1] range.

    Returns:
        Depth array of shape [H, W] with values in [0, 1] if normalized,
        or original scale otherwise.
    """
    z_channel = point_map[..., 2]
    z_depth = np.where(mask, z_channel, 0.0)
    if normalize:
        valid = z_depth[mask]
        if len(valid) > 0:
            vmin, vmax = np.percentile(valid, (1, 99))
            if vmax > vmin:
                z_depth = np.clip(z_depth, vmin, vmax)
                z_depth = (z_depth - vmin) / (vmax - vmin)
    return z_depth.astype(np.float32)


def depth_to_uint16(depth: np.ndarray, invert: bool = False) -> np.ndarray:
    """Converts depth to 16-bit unsigned integer.

    Args:
        depth: Depth array with values in [0, 1].
        invert: Whether to invert values before conversion.

    Returns:
        uint16 array with values in [0, 65535].
    """
    if invert:
        depth = 1.0 - depth
    uint16_img = (depth * 65535).astype(np.uint16)
    return uint16_img


class Extractor:
    """Extractor for converting NPZ segments to disparity PNGs.

    This class handles loading NPZ files, extracting the depth channel,
    and saving 16-bit PNG files organized by segment.

    Attributes:
        config: Configuration for the extractor.

    Example:
        >>> config = ExtractorConfig(
        ...     segments_folder="workspace/segments",
        ...     output_folder="workspace/temp_frames",
        ... )
        >>> extractor = Extractor(config)
        >>> paths = extractor.extract_all_segments()
    """

    def __init__(self, config: ExtractorConfig):
        """Initializes the Extractor.

        Args:
            config: Configuration for the extractor.
        """
        self.config = config

    def _get_segment_files(self) -> List[Path]:
        """Finds all NPZ segment files in the segments folder.

        Returns:
            Sorted list of paths to NPZ files.
        """
        folder = Path(self.config.segments_folder)
        files = sorted(folder.glob(f"{self.config.file_prefix}*{self.config.file_ext}"))
        return files

    def extract_segment(self, segment_index: int) -> Path:
        """Extracts frames from a single segment.

        Args:
            segment_index: Index of the segment to extract.

        Returns:
            Path to the output directory containing PNG frames.

        Raises:
            ValueError: If segment_index is out of range.
        """
        files = self._get_segment_files()
        if segment_index >= len(files):
            raise ValueError(f"Segment index {segment_index} out of range (0-{len(files) - 1})")

        npz_path = files[segment_index]
        point_map, mask, metadata = load_npz_segment(npz_path)

        segment_name = f"part_{segment_index:04d}"
        output_dir = Path(self.config.output_folder) / segment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        num_frames = metadata["frame_count"]
        for i in range(num_frames):
            depth = extract_depth_from_point_map(point_map[i], mask[i], normalize=self.config.normalize)
            uint16 = depth_to_uint16(depth, invert=self.config.invert)
            frame_filename = f"frame_{i:04d}.png"
            frame_path = output_dir / frame_filename
            Image.fromarray(uint16).save(frame_path, bits=16)

        print(f"Extracted {num_frames} frames to {output_dir}")
        return output_dir

    def extract_all_segments(self) -> List[Path]:
        """Extracts frames from all segments.

        Returns:
            List of paths to output directories.
        """
        files = self._get_segment_files()
        output_dirs = []

        for i in range(len(files)):
            path = self.extract_segment(i)
            output_dirs.append(path)

        print(f"Extracted {len(output_dirs)} segments")
        return output_dirs


def extract_disparity_frames(
    segments_folder: str, output_folder: str, normalize: bool = True, invert: bool = False
) -> List[Path]:
    """High-level function to extract disparity frames from all segments.

    This is the main entry point for Phase 3 of the GC-SEG pipeline.

    Args:
        segments_folder: Directory containing NPZ segment files.
        output_folder: Directory to save PNG frames.
        normalize: Whether to normalize depth values (default: True).
        invert: Whether to invert depth values (default: False).

    Returns:
        List of paths to output directories.

    Example:
        >>> from gc_seg import extract_disparity_frames
        >>> paths = extract_disparity_frames(
        ...     segments_folder="workspace/segments",
        ...     output_folder="workspace/temp_frames",
        ... )
    """
    config = ExtractorConfig(
        segments_folder=segments_folder, output_folder=output_folder, normalize=normalize, invert=invert
    )
    extractor = Extractor(config)
    return extractor.extract_all_segments()


def extract_single_segment(
    segment_index: int, segments_folder: str, output_folder: str, normalize: bool = True, invert: bool = False
) -> Path:
    """Extracts frames from a single segment.

    Args:
        segment_index: Index of the segment to extract.
        segments_folder: Directory containing NPZ segment files.
        output_folder: Directory to save PNG frames.
        normalize: Whether to normalize depth values.
        invert: Whether to invert depth values.

    Returns:
        Path to the output directory containing PNG frames.
    """
    config = ExtractorConfig(
        segments_folder=segments_folder, output_folder=output_folder, normalize=normalize, invert=invert
    )
    extractor = Extractor(config)
    return extractor.extract_segment(segment_index)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract disparity PNGs from NPZ segments")
    parser.add_argument(
        "--segments-folder", "-i", default="workspace/segments", help="Folder containing NPZ segment files"
    )
    parser.add_argument("--output-folder", "-o", default="workspace/temp_frames", help="Output folder for PNG frames")
    parser.add_argument(
        "--segment-index", "-s", type=int, default=None, help="Extract single segment by index (default: all segments)"
    )
    parser.add_argument("--no-normalize", action="store_true", help="Disable normalization")
    parser.add_argument("--invert", action="store_true", help="Invert depth values")

    args = parser.parse_args()

    if args.segment_index is not None:
        extract_single_segment(
            args.segment_index,
            args.segments_folder,
            args.output_folder,
            normalize=not args.no_normalize,
            invert=args.invert,
        )
    else:
        extract_disparity_frames(
            args.segments_folder, args.output_folder, normalize=not args.no_normalize, invert=args.invert
        )
