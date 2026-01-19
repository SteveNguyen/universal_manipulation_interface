#!/usr/bin/env python3
"""
Downsample high-resolution videos for SLAM processing while keeping originals.
This allows using 4K videos for training but faster resolution for SLAM.

Usage: python scripts/downsample_video_for_slam.py <video_path> --target_width 2704
"""

import click
import pathlib
import subprocess
import shutil


@click.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--target_width', type=int, default=2704,
              help='Target width in pixels (height will be calculated to maintain aspect ratio)')
@click.option('--output_suffix', type=str, default='_slam',
              help='Suffix to add to output filename')
@click.option('--backup_original', is_flag=True, default=True,
              help='Backup original video with _original suffix')
def main(video_path, target_width, output_suffix, backup_original):
    """
    Downsample a video for SLAM processing while optionally backing up the original.

    Example:
        python scripts/downsample_video_for_slam.py demo_video/raw_video.mp4 --target_width 2704

    This will:
    1. Backup raw_video.mp4 to raw_video_original.mp4
    2. Downsample to 2704 width and save as raw_video_slam.mp4
    3. Rename raw_video_slam.mp4 to raw_video.mp4 (for SLAM to use)
    """

    video_path = pathlib.Path(video_path)

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    # Get video dimensions
    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        str(video_path)
    ]

    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split(','))
        print(f"Original resolution: {width}x{height}")
    except Exception as e:
        print(f"Error getting video dimensions: {e}")
        return 1

    # Calculate target height maintaining aspect ratio
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)

    # Round to even numbers (required for some codecs)
    target_height = target_height - (target_height % 2)

    print(f"Target resolution: {target_width}x{target_height}")

    # Check if downsampling is needed
    if width <= target_width:
        print(f"Video is already at or below target width ({width} <= {target_width})")
        print("No downsampling needed.")
        return 0

    # Create output path
    output_path = video_path.parent / f"{video_path.stem}{output_suffix}{video_path.suffix}"

    # Downsample using ffmpeg with high-quality settings
    print(f"Downsampling to {target_width}x{target_height}...")
    print("This may take a few minutes...")

    ffmpeg_cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', f'scale={target_width}:{target_height}:flags=lanczos',
        '-c:v', 'libx264',
        '-crf', '18',  # High quality (lower = better, 18 is near-lossless)
        '-preset', 'slow',  # Better compression
        '-c:a', 'copy',  # Copy audio without re-encoding
        '-movflags', '+faststart',  # Enable fast start for streaming
        '-y',  # Overwrite output file
        str(output_path)
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"✓ Downsampled video saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during downsampling: {e}")
        return 1

    # Backup original if requested
    if backup_original:
        backup_path = video_path.parent / f"{video_path.stem}_original{video_path.suffix}"
        if not backup_path.exists():
            print(f"Backing up original to: {backup_path}")
            shutil.move(video_path, backup_path)
            print(f"✓ Original backed up")
        else:
            print(f"Backup already exists: {backup_path}")
            video_path.unlink()

        # Rename downsampled version to original name (for SLAM to use)
        shutil.move(output_path, video_path)
        print(f"✓ Downsampled version renamed to: {video_path}")
        print(f"\nFor training, use the original: {backup_path}")
        print(f"For SLAM, use the downsampled: {video_path}")
    else:
        print(f"\nOriginal: {video_path}")
        print(f"Downsampled: {output_path}")

    print("\n✓ Done!")
    return 0


if __name__ == '__main__':
    exit(main())
