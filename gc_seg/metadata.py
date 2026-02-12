"""Metadata and segment mapping utilities for GC-SEG pipeline.

This module provides functionality for video metadata probing and segment mapping
to support segmented processing of long videos.

Example:
    >>> from gc_seg import create_segment_mapping
    >>> mapping = create_segment_mapping("video.mp4", window_size=110, overlap=25)
    >>> print(f"Created {len(mapping.segments)} segments")
"""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file.

    Attributes:
        path: Path to the video file.
        num_frames: Total number of frames in the video.
        fps: Frames per second (calculated from time_base).
        time_base: Tuple of (numerator, denominator) for time base.
        width: Video width in pixels.
        height: Video height in pixels.
        duration: Video duration in seconds.
        video_index: Index of the video stream (default 0).
    """

    path: str
    num_frames: int
    fps: float
    time_base: Tuple[int, int]
    width: int
    height: int
    duration: float
    video_index: int = 0


@dataclass
class Segment:
    """Represents a contiguous segment of video frames.

    Attributes:
        name: Name of the segment (e.g., 'Segment_001').
        start_frame: Zero-indexed start frame number.
        end_frame: Zero-indexed end frame number (inclusive).
        start_time: Start time in seconds.
        end_time: End time in seconds.
    """

    name: str
    start_frame: int
    end_frame: int
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def frame_count(self) -> int:
        """Returns the number of frames in this segment."""
        return self.end_frame - self.start_frame + 1

    @property
    def frame_range(self) -> List[int]:
        """Returns a list of all frame indices in this segment."""
        return list(range(self.start_frame, self.end_frame + 1))


@dataclass
class SegmentMapping:
    """Maps a video to its processing segments.

    Attributes:
        video_path: Path to the source video file.
        metadata: Video metadata including fps, resolution, etc.
        segments: List of Segment objects.
        overlap: Number of overlapping frames between segments.
    """

    video_path: str
    metadata: VideoMetadata
    segments: List[Segment] = field(default_factory=list)
    overlap: int = 25

    def add_segment(self, name: str, start_frame: int, end_frame: int) -> Segment:
        """Adds a segment to the mapping.

        Args:
            name: Name for the segment.
            start_frame: Zero-indexed start frame.
            end_frame: Zero-indexed end frame (inclusive).

        Returns:
            The newly created Segment object.
        """
        segment = Segment(
            name=name,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=start_frame / self.metadata.fps,
            end_time=end_frame / self.metadata.fps,
        )
        self.segments.append(segment)
        return segment

    def auto_segment(self, window_size: int = 110, overlap: Optional[int] = None) -> List[Segment]:
        """Automatically creates segments based on window size and overlap.

        Args:
            window_size: Number of frames per segment.
            overlap: Number of overlapping frames between segments.
                If None, uses the instance default.

        Returns:
            List of created Segment objects.

        Raises:
            ValueError: If window_size <= 0 or overlap < 0.
        """
        if overlap is None:
            overlap = self.overlap

        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")

        self.overlap = overlap
        self.segments.clear()

        total_frames = self.metadata.num_frames
        stride = window_size - overlap

        segment_num = 1
        start = 0

        while start < total_frames:
            end = min(start + window_size - 1, total_frames - 1)
            segment_frames = end - start + 1

            if segment_num > 1 and segment_frames <= overlap:
                break

            self.add_segment(name=f"Segment_{segment_num:03d}", start_frame=start, end_frame=end)
            start += stride
            segment_num += 1

        return self.segments

    def to_dict(self) -> Dict:
        """Converts the mapping to a dictionary.

        Returns:
            Dictionary representation of the mapping.
        """
        return {
            "video_path": self.video_path,
            "metadata": {
                "num_frames": self.metadata.num_frames,
                "fps": self.metadata.fps,
                "time_base": f"{self.metadata.time_base[0]}/{self.metadata.time_base[1]}",
                "width": self.metadata.width,
                "height": self.metadata.height,
                "duration": self.metadata.duration,
            },
            "overlap": self.overlap,
            "segments": [
                {
                    "name": s.name,
                    "start_frame": s.start_frame,
                    "end_frame": s.end_frame,
                    "frame_count": s.frame_count,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                }
                for s in self.segments
            ],
        }

    def to_json(self, path: Optional[Path] = None) -> str:
        """Converts the mapping to a JSON string.

        Args:
            path: Optional path to save the JSON file.

        Returns:
            JSON string representation.
        """
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)
        if path:
            Path(path).write_text(json_str)
        return json_str

    @classmethod
    def from_json(cls, path: Path, video_path: str) -> "SegmentMapping":
        """Creates a SegmentMapping from a JSON file.

        Args:
            path: Path to the JSON file.
            video_path: Path to the video file (used for validation).

        Returns:
            A new SegmentMapping instance.
        """
        data = json.loads(Path(path).read_text())
        meta = data["metadata"]
        time_base = meta["time_base"].split("/")

        metadata = VideoMetadata(
            path=video_path,
            num_frames=meta["num_frames"],
            fps=meta["fps"],
            time_base=(int(time_base[0]), int(time_base[1])),
            width=meta["width"],
            height=meta["height"],
            duration=meta["duration"],
        )

        mapping = cls(video_path=video_path, metadata=metadata, overlap=data["overlap"])
        for s in data["segments"]:
            mapping.add_segment(s["name"], s["start_frame"], s["end_frame"])

        return mapping


def probe_video_timebase(video_path: str) -> Tuple[int, int]:
    """Probes the video file for its time base using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Tuple of (numerator, denominator) representing the time base.

    Raises:
        RuntimeError: If ffprobe fails to read the video.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=time_base",
        "-of",
        "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    time_base_str = result.stdout.strip().split("/")
    if len(time_base_str) == 2:
        return (int(time_base_str[0]), int(time_base_str[1]))
    return (1, 1)


def probe_video_metadata(video_path: str) -> VideoMetadata:
    """Probes the video file for metadata using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        VideoMetadata object with the probed information.

    Raises:
        RuntimeError: If ffprobe fails to read the video.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,duration,width,height,r_frame_rate",
        "-of",
        "json=c=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    stream = data["streams"][0]

    num_frames = int(stream.get("nb_read_frames", 0))
    duration = float(stream.get("duration", 0))
    width = int(stream["width"])
    height = int(stream["height"])

    fps = 0.0
    time_base = (1, 1)

    r_frame_rate = stream.get("r_frame_rate", "0/1")
    if "/" in r_frame_rate:
        num, denom = r_frame_rate.split("/")
        if int(denom) > 0:
            fps = int(num) / int(denom)
            time_base = (int(denom), int(num))
    else:
        fps = float(r_frame_rate)

    if num_frames == 0 and duration > 0:
        num_frames = int(duration * fps)

    return VideoMetadata(
        path=video_path,
        num_frames=num_frames,
        fps=fps,
        time_base=time_base,
        width=width,
        height=height,
        duration=duration,
    )


def create_segment_mapping(
    video_path: str, window_size: int = 110, overlap: int = 25, use_exact_timebase: bool = True
) -> SegmentMapping:
    """Creates a complete segment mapping for a video.

    This is the main entry point for Phase 1 of the GC-SEG pipeline.

    Args:
        video_path: Path to the input video file.
        window_size: Number of frames per segment (default 110).
        overlap: Number of overlapping frames between segments (default 25).
        use_exact_timebase: If True, use ffprobe to get exact time base.

    Returns:
        A SegmentMapping object with auto-generated segments.

    Example:
        >>> mapping = create_segment_mapping("video.mp4", window_size=50, overlap=10)
        >>> print(f"Created {len(mapping.segments)} segments")
    """
    if use_exact_timebase:
        try:
            time_base = probe_video_timebase(video_path)
        except Exception:
            time_base = (1, 1)
    else:
        time_base = (1, 1)

    metadata = probe_video_metadata(video_path)
    metadata.time_base = time_base

    mapping = SegmentMapping(video_path=video_path, metadata=metadata, overlap=overlap)

    mapping.auto_segment(window_size=window_size, overlap=overlap)

    return mapping


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video metadata and segment mapping")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--window-size", type=int, default=110, help="Frames per segment")
    parser.add_argument("--overlap", type=int, default=25, help="Overlap between segments")
    parser.add_argument("--output", "-o", help="Output JSON path")

    args = parser.parse_args()

    mapping = create_segment_mapping(args.video_path, window_size=args.window_size, overlap=args.overlap)

    print(mapping.to_json(args.output))
