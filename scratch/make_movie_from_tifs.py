import tifffile
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os
import sys
from tqdm import tqdm
from skimage.transform import resize

def normalize_to_uint8(frame):
    """Normalize frame to uint8 (0-255) range."""
    if frame.dtype != np.uint8:
        frame = frame.astype(float)
        frame = ((frame - frame.min()) * 255 / (frame.max() - frame.min())).astype(np.uint8)
    return frame

def process_tif_files(tif_folder, scale_factor=1.0):
    """
    Process TIF files and return normalized frames.
    
    Args:
        tif_folder: Path to folder containing TIF files
        scale_factor: Scale factor for frame size (default: 1.0)
    
    Returns:
        list of processed frames
    """
    # get all tif files in the folder
    tif_files = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]
    tif_files.sort()
    
    print(f"Found {len(tif_files)} TIF files")
    
    # Get frame dimensions from first file
    with tifffile.TiffFile(os.path.join(tif_folder, tif_files[0])) as tif:
        first_frame = tif.asarray()
        new_height = int(first_frame.shape[0] * scale_factor)
        new_width = int(first_frame.shape[1] * scale_factor)
    
    # Prepare frames
    frames = []
    print("Processing frames...")
    for tif_file in tqdm(tif_files):
        with tifffile.TiffFile(os.path.join(tif_folder, tif_file)) as tif:
            frame = tif.asarray()
            
            # Convert to uint8 if needed
            frame = normalize_to_uint8(frame)
            
            # Resize if scale_factor != 1.0
            if scale_factor != 1.0:
                frame = resize(frame, (new_height, new_width), 
                             preserve_range=True).astype(np.uint8)
            
            frames.append(frame)
    
    return np.array(frames)

def save_mp4(frames, output_file, fps=30, quality=7):
    """
    Save frames as MP4 video.
    
    Args:
        frames: numpy array of frames
        output_file: Path to output MP4 file
        fps: Frames per second (default: 30)
        quality: Compression quality (0-10, default: 7)
    """
    print("Writing MP4 file...")
    
    # Calculate bitrate based on quality (higher quality = higher bitrate)
    bitrate = int(1000 + (400 * quality))  # 1000k-5000k range
    
    iio.imwrite(
        output_file,
        frames,
        plugin="FFMPEG",
        fps=fps,
        codec='libx264',
        bitrate=f"{bitrate}k",
        output_params=["-preset", "medium"],
        macro_block_size=1,
        pixelformat='yuv420p'
    )
    
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"MP4 saved to: {output_file}")
    print(f"File size: {size_mb:.1f} MB")

def save_gif(frames, output_file, fps=10):
    """
    Save frames as GIF.
    
    Args:
        frames: numpy array of frames
        output_file: Path to output GIF file
        fps: Frames per second (default: 10)
    """
    print("Writing GIF file...")
    
    # For GIF, we use a lower default FPS as GIFs are typically slower
    duration = 1000 / fps  # Convert fps to milliseconds per frame
    
    iio.imwrite(
        output_file,
        frames,
        plugin="pillow",
        duration=duration,
        loop=0  # 0 means loop forever
    )
    
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"GIF saved to: {output_file}")
    print(f"File size: {size_mb:.1f} MB")

def convert_tifs_to_video(tif_folder, output_name, make_mp4=True, make_gif=True, 
                         mp4_fps=30, gif_fps=10, quality=7, scale_factor=1.0):
    """
    Convert TIF files to MP4 and/or GIF.
    
    Args:
        tif_folder: Path to folder containing TIF files
        output_name: Base name for output files (without extension)
        make_mp4: Whether to create MP4 output (default: True)
        make_gif: Whether to create GIF output (default: True)
        mp4_fps: Frames per second for MP4 (default: 30)
        gif_fps: Frames per second for GIF (default: 10)
        quality: Compression quality for MP4 (0-10, default: 7)
        scale_factor: Scale factor for frame size (default: 1.0)
    """
    # Process frames
    frames = process_tif_files(tif_folder, scale_factor)
    
    # Save MP4 if requested
    if make_mp4:
        mp4_file = f"{output_name}.mp4"
        save_mp4(frames, mp4_file, mp4_fps, quality)
    
    # Save GIF if requested
    if make_gif:
        gif_file = f"{output_name}.gif"
        save_gif(frames, gif_file, gif_fps)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python make_movie_from_tifs.py <tif_folder> <output_name> "
              "[--no-mp4] [--no-gif] [--mp4-fps FPS] [--gif-fps FPS] "
              "[--quality QUALITY] [--scale SCALE]")
        sys.exit(1)
    
    import argparse
    parser = argparse.ArgumentParser(description='Convert TIF files to MP4 and/or GIF')
    parser.add_argument('tif_folder', help='Folder containing TIF files')
    parser.add_argument('output_name', help='Base name for output files (without extension)')
    parser.add_argument('--no-mp4', action='store_true', help='Skip MP4 creation')
    parser.add_argument('--no-gif', action='store_true', help='Skip GIF creation')
    parser.add_argument('--mp4-fps', type=int, default=30, help='Frames per second for MP4')
    parser.add_argument('--gif-fps', type=int, default=10, help='Frames per second for GIF')
    parser.add_argument('--quality', type=int, default=7, help='MP4 quality (0-10)')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for frame size')
    
    args = parser.parse_args()
    
    convert_tifs_to_video(
        args.tif_folder,
        args.output_name,
        make_mp4=not args.no_mp4,
        make_gif=not args.no_gif,
        mp4_fps=args.mp4_fps,
        gif_fps=args.gif_fps,
        quality=args.quality,
        scale_factor=args.scale
    ) 