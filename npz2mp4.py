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
    mode: str = 'bilateral', # 'bilateral', 'guided', or 'raw'
    radius: int = 7,         # For Bilateral, this is 'd' (diameter)
    sigma_color: int = 75,   # Higher = ignore more texture (Anti-Hair)
    sigma_space: int = 75,   # Higher = smoother gradients
    eps: float = 0.1,        # Only for 'guided' mode
    crf: int = 12,
    fps_override: float = None
):
    """
    Args:
        mode (str): 
            'bilateral': Sharp edges, smooth surfaces (Best for people).
            'guided': Maximum detail, but can look 'hairy' or like stone.
            'raw': No smart filtering, just a standard blurry upscale.
    """
    assert mode in ['bilateral', 'guided', 'raw'], "Mode must be 'bilateral', 'guided', or 'raw'"
    
    # 1. Setup Paths
    base_name = os.path.splitext(input_npz)[0]
    final_output = output_path or f"{base_name}_{mode}.mp4"
    temp_dir = f"temp_{mode}_{os.path.basename(base_name)}"
    
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    print(f"Loading NPZ: {input_npz}")
    data = np.load(input_npz)
    disp_seq = data['disparity'] if 'disparity' in data else 1.0 / (data['point_map'][..., 2] + 1e-6)
    
    reader = imageio.get_reader(guide_video)
    fps = fps_override or reader.get_meta_data()['fps']

    print(f"Processing {len(disp_seq)} frames using {mode} upscale...")
    for i, low_res_frame in enumerate(tqdm(disp_seq)):
        try:
            high_res_color = reader.get_data(i)
        except IndexError: break

        h, w = high_res_color.shape[:2]
        # Normalize and cast to float32 for OpenCV
        low_res_norm = (low_res_frame.astype(np.float32) - low_res_frame.min()) / (low_res_frame.max() - low_res_frame.min() + 1e-6)
        coarse_depth = cv2.resize(low_res_norm, (w, h), interpolation=cv2.INTER_LINEAR)

        if mode == 'bilateral':
            # 1. Normalize guide to 0-255 float32 (standard for JBF)
            guide_img = high_res_color.astype(np.float32)
            
            # 2. Ensure coarse_depth is float32
            src_img = coarse_depth.astype(np.float32)

            # 3. Apply the filter using positional arguments
            processed_depth = cv2.ximgproc.jointBilateralFilter(
                guide_img, 
                src_img, 
                radius, 
                sigma_color, 
                sigma_space
            )
        elif mode == 'guided':
            # Guided Filter: Tries to match the guide image's gradients exactly
            guide_gray = cv2.cvtColor(high_res_color, cv2.COLOR_RGB2GRAY)
            processed_depth = cv2.ximgproc.guidedFilter(
                guide=guide_gray, src=coarse_depth, radius=radius, eps=eps
            )
        else:
            # Raw mode: No smart filtering
            processed_depth = coarse_depth

        # Save 16-bit PNG
        depth_16bit = (np.clip(processed_depth, 0, 1) * 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(temp_dir, f"frame_{i:04d}.png"), depth_16bit)

    # 3. FFmpeg Command
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-hide_banner', '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%04d.png'),
        '-c:v', 'libx265', '-crf', str(crf), '-preset', 'veryslow',
        '-pix_fmt', 'yuv420p10le',
        '-x265-params', 'aq-mode=3:no-sao=1:strong-intra-smoothing=1',
        final_output
    ]
    subprocess.run(ffmpeg_cmd)
    shutil.rmtree(temp_dir)
    reader.close()
    print(f"SUCCESS: {final_output}")

if __name__ == "__main__":
    Fire(convert)