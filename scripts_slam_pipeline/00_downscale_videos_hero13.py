"""
Downscale Hero 13 4K videos to 2.7K for UMI pipeline.

Usage:
    python scripts_slam_pipeline/00_downscale_videos_hero13.py data/dataset_redcube2

This script should be run BEFORE run_slam_pipeline.py for Hero 13 datasets.
It downscales all MP4 videos from 4K to 2.7K (2704x2028) resolution.

IMPORTANT: This script preserves the original 4K files as *_4k_original.MP4
because the IMU extraction needs to happen from the original files
(GPMF metadata may not survive re-encoding).

The pipeline will:
1. First run IMU extraction on *_4k_original.MP4 files if they exist
2. Then use the downscaled 2.7K files for SLAM processing
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import subprocess
from tqdm import tqdm
import shutil


@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-w', '--width', type=int, default=2704, help='Target width')
@click.option('-ht', '--height', type=int, default=2028, help='Target height')
@click.option('-f', '--force', is_flag=True, default=False, help='Re-encode even if already correct resolution')
@click.option('--delete-originals', is_flag=True, default=False, help='Delete original 4K files after downscaling')
def main(session_dir, width, height, force, delete_originals):
    """Downscale Hero 13 4K videos to 2.7K resolution."""

    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()

        # Find all MP4 files (excluding already processed)
        mp4_files = list(session.glob('*.MP4')) + list(session.glob('*.mp4'))
        mp4_files = [f for f in mp4_files if '_4k_original' not in f.stem
                     and '_2.7k' not in f.stem
                     and not f.is_symlink()]

        print(f"Found {len(mp4_files)} MP4 files in {session}")

        if not mp4_files:
            print("No MP4 files to process!")
            continue

        processed = 0
        skipped = 0

        for mp4_path in tqdm(mp4_files, desc="Processing videos"):
            # Check current resolution
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0',
                str(mp4_path)
            ]

            try:
                result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                current_w, current_h = map(int, result.stdout.strip().split(','))
            except (subprocess.CalledProcessError, ValueError) as e:
                print(f"\nWarning: Could not probe {mp4_path.name}: {e}")
                continue

            # Check if already correct resolution
            if current_w == width and current_h == height:
                if not force:
                    skipped += 1
                    continue

            # Check if 4K (3840x2880 or similar)
            if current_w < 3000:
                print(f"\n  {mp4_path.name}: {current_w}x{current_h} - not 4K, skipping")
                skipped += 1
                continue

            tqdm.write(f"  {mp4_path.name}: {current_w}x{current_h} -> {width}x{height}")

            # Paths
            original_backup = mp4_path.with_stem(mp4_path.stem + '_4k_original')
            output_path = mp4_path  # Downscaled file replaces original name

            # First, rename original to backup
            if not original_backup.exists():
                shutil.move(str(mp4_path), str(original_backup))
            else:
                # Backup already exists, use it as source
                pass

            # Downscale with ffmpeg
            # Note: We copy the data streams but they may not be perfectly synced after re-encoding
            # Note: We don't copy data streams (GPMF/timecode) because:
            # 1. The timecode stream (tmcd) has "codec none" which FFmpeg can't write to MP4
            # 2. IMU extraction uses the original *_4k_original files anyway
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
                '-i', str(original_backup),
                '-vf', f'scale={width}:{height}',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-c:a', 'copy',
                '-map', '0:v',  # Video
                '-map', '0:a?',  # Audio (optional)
                '-movflags', '+faststart',
                str(output_path)
            ]

            try:
                result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

                # Check if output file is valid (non-zero size)
                if output_path.stat().st_size == 0:
                    raise RuntimeError(f"FFmpeg produced 0-byte file. Stderr: {result.stderr.decode()[:500] if result.stderr else 'none'}")

                processed += 1

                if delete_originals:
                    original_backup.unlink()
                    tqdm.write(f"    Deleted original")

            except (subprocess.CalledProcessError, RuntimeError) as e:
                if hasattr(e, 'stderr') and e.stderr:
                    tqdm.write(f"    Error: {e.stderr.decode()[:300]}")
                else:
                    tqdm.write(f"    Error: {e}")
                # Restore original if downscaling failed
                if original_backup.exists():
                    if output_path.exists():
                        output_path.unlink()  # Remove failed output
                    shutil.move(str(original_backup), str(mp4_path))

        print(f"\n{'='*60}")
        print(f"Done! Processed: {processed}, Skipped: {skipped}")
        print(f"{'='*60}")

        if not delete_originals and processed > 0:
            print(f"\nOriginal 4K files backed up with '_4k_original' suffix")
            print("These are needed for IMU extraction!")
            print("\nNext steps:")
            print(f"  1. Run the pipeline with Hero 13 mode:")
            print(f"     uv run python run_slam_pipeline.py --camera_type hero13 -c example/calibration {session}")


if __name__ == '__main__':
    main()
