import numpy as np
import cv2
import os
import subprocess
import shutil
import imageio
from tqdm import tqdm
from fire import Fire

def convert(
    input_npz: str,
    guide_video: str,
    output_path: str = None,
    radius: int = 12,
    eps: float = 0.01,
    crf: int = 12,
    fps_override: float = None,
    keep_temp: bool = False
):
    """
    Converts GeometryCrafter .npz data to 10-bit x265 video using Guided Filter Upscaling.

    Args:
        input_npz (str): Path to the .npz file.
        guide_video (str): Path to the original high-res video.
        output_path (str): Path for the output mp4. If None, uses input_npz name.
        radius (int): Guided filter radius. Higher = smoother, Lower = more local.
        eps (float): Guided filter epsilon. Lower = sharper edges.
        crf (int): FFmpeg Constant Rate Factor (0-51). 12 is very high quality.
        fps_override (float): Manually set FPS. If None, detects from guide_video.
        keep_temp (bool): If True, does not delete the 16-bit PNG frames.
    """
    
    if output_path is None:
        base_name = os.path.splitext(input_npz)[0]
        output_path = f"{base_name}_GC_depth.mp4"

    temp_dir = f"temp_{os.path.basename(input_npz).split('.')[0]}"
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    print(f"Loading NPZ: {input_npz}")
    data = np.load(input_npz)
    disp_seq = data['disparity'] if 'disparity' in data else 1.0 / (data['point_map'][..., 2] + 1e-6)

    print(f"Opening Guide Video: {guide_video}")
    reader = imageio.get_reader(guide_video)
    fps = fps_override or reader.get_meta_data()['fps']

    print(f"Processing {len(disp_seq)} frames...")

    for i, low_res_frame in enumerate(tqdm(disp_seq)):
        try:
            high_res_color = reader.get_data(i)
        except IndexError:
            break

        guide_gray = cv2.cvtColor(high_res_color, cv2.COLOR_RGB2GRAY)
        h, w = guide_gray.shape

        # Normalization
        d_min, d_max = low_res_frame.min(), low_res_frame.max()
        low_res_norm = (low_res_frame - d_min) / (d_max - d_min + 1e-6)
        
        coarse_depth = cv2.resize(low_res_norm, (w, h), interpolation=cv2.INTER_LINEAR)

        # Guided Filter
        smart_depth = cv2.ximgproc.guidedFilter(
            guide=guide_gray, src=coarse_depth, radius=radius, eps=eps
        )

        # 16-bit Conversion
        depth_16bit = (np.clip(smart_depth, 0, 1) * 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(temp_dir, f"frame_{i:04d}.png"), depth_16bit)

    reader.close()

    # FFmpeg Command
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-hide_banner',
        '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%04d.png'),
        '-c:v', 'libx265',
        '-crf', str(crf),            
        '-preset', 'veryslow',   
        '-pix_fmt', 'yuv420p10le',
        '-x265-params', 'aq-mode=3:no-sao=1:strong-intra-smoothing=1',
        output_path
    ]

    print("\nRunning FFmpeg...")
    subprocess.run(ffmpeg_cmd)

    if not keep_temp:
        shutil.rmtree(temp_dir)

    print(f"DONE! Saved to {output_path}")

if __name__ == "__main__":
    Fire(convert)