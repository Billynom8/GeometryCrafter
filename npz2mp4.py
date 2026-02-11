import numpy as np
import cv2
import os
import subprocess
import shutil

# --- CONFIG ---
input_file = "workspace/output/video1.npz"
output_file = "geometry_16bit_clean.mp4"
temp_dir = "temp_depth_frames"

if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir)

print(f"Loading {input_file}...")
data = np.load(input_file)
disp_seq = data['disparity'] if 'disparity' in data else 1.0 / (data['point_map'][..., 2] + 1e-6)

# Normalize to 16-bit
disp_min, disp_max = disp_seq.min(), disp_seq.max()
disp_normalized = (disp_seq - disp_min) / (disp_max - disp_min)
disp_16bit = (disp_normalized * 65535).astype(np.uint16)

print(f"Saving {len(disp_16bit)} 16-bit PNG frames...")
for i, frame in enumerate(disp_16bit):
    # cv2.imwrite handles uint16 PNGs perfectly
    cv2.imwrite(os.path.join(temp_dir, f"frame_{i:04d}.png"), frame)
    if i % 20 == 0:
        print(f"Saved frame {i}", end='\r')

# --- THE FFMPEG COMMAND ---
# This uses the PNG sequence as input. 
# It tells FFmpeg to interpret the 16-bit PNGs and output 10-bit HEVC.
ffmpeg_cmd = [
    'ffmpeg', '-y',
    '-framerate', '25',
    '-i', os.path.join(temp_dir, 'frame_%04d.png'),
    '-c:v', 'libx265',
    '-crf', '12',            # Extremely high quality
    '-preset', 'veryslow',   # Best compression for smooth gradients
    '-pix_fmt', 'yuv420p10le',
    '-x265-params', 'aq-mode=3:no-sao=1:strong-intra-smoothing=1', # Anti-banding tweaks
    output_file
]

print("\nRunning FFmpeg x265 10-bit conversion...")
subprocess.run(ffmpeg_cmd)

# Cleanup
print(f"Cleaning up temp files...")
shutil.rmtree(temp_dir)

print(f"DONE! Output saved to {output_file}")