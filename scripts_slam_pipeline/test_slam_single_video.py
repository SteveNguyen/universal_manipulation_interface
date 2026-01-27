"""
Test SLAM on a single video file.

This script is useful for quick testing of the SLAM pipeline on individual videos
without running the full UMI pipeline.

Usage:
    # Test with Hero 13 video (2.7K resolution)
    uv run python scripts_slam_pipeline/test_slam_single_video.py video.mp4 --camera_type hero13

    # Test with GoPro 9/10/11 video
    uv run python scripts_slam_pipeline/test_slam_single_video.py video.mp4

    # Use existing map for localization
    uv run python scripts_slam_pipeline/test_slam_single_video.py video.mp4 --load_map map_atlas.osa

    # Skip Docker pull (faster if image already cached)
    uv run python scripts_slam_pipeline/test_slam_single_video.py video.mp4 --no-docker-pull
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import subprocess
import tempfile
import shutil
import json
import numpy as np
import cv2
import av
from umi.common.cv_util import draw_predefined_mask, draw_predefined_mask_hero13


def extract_imu_data(video_path, output_dir, docker_image="chicheng/openicc:latest",
                     no_docker_pull=False):
    """Extract IMU data from GoPro video using Docker container.

    Args:
        video_path: Path to video with GPMF metadata (original, not re-encoded)
        output_dir: Directory to save imu_data.json
        docker_image: Docker image for IMU extraction
        no_docker_pull: Skip pulling Docker image
    """
    video_path = pathlib.Path(video_path).absolute()
    source_video = video_path
    print(f"  Using video for IMU: {video_path.name}")

    # Pull Docker image
    if not no_docker_pull:
        cmd = ['docker', 'pull', docker_image]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f"  Warning: Docker pull failed: {result.stderr.decode()[:200]}")

    # Run IMU extraction via Docker
    # Mount the SOURCE video directory (not output dir) so we can read the original file
    source_dir = source_video.parent
    mount_target = pathlib.Path('/data')

    # Create temp symlink or use the source filename
    video_mount = mount_target / source_video.name
    json_mount = mount_target / 'imu_data.json'

    cmd = [
        'docker', 'run', '--rm',
        '--volume', f'{source_dir}:/data',
        docker_image,
        'node',
        '/OpenImuCameraCalibrator/javascript/extract_metadata_single.js',
        str(video_mount),
        str(json_mount)
    ]

    stdout_path = output_dir / 'extract_imu_stdout.txt'
    stderr_path = output_dir / 'extract_imu_stderr.txt'

    result = subprocess.run(
        cmd,
        stdout=stdout_path.open('w'),
        stderr=stderr_path.open('w')
    )

    # IMU data is written to source_dir, copy to output_dir
    imu_source = source_dir / 'imu_data.json'
    imu_dest = output_dir / 'imu_data.json'

    if imu_source.exists() and imu_source.stat().st_size > 0:
        # Copy to output dir if different
        if imu_source != imu_dest:
            shutil.copy(str(imu_source), str(imu_dest))

        with open(imu_dest) as f:
            imu_data = json.load(f)

        # Count samples - handle different JSON formats
        n_samples = 0
        if 'timestamps' in imu_data:
            n_samples = len(imu_data['timestamps'])
        elif '1' in imu_data and 'streams' in imu_data['1']:
            # Hero 13 format: {"1": {"streams": {"ACCL": {"samples": [...]}, "GYRO": ...}}}
            streams = imu_data['1']['streams']
            if 'ACCL' in streams and 'samples' in streams['ACCL']:
                n_samples = len(streams['ACCL']['samples'])
            elif 'GYRO' in streams and 'samples' in streams['GYRO']:
                n_samples = len(streams['GYRO']['samples'])

        print(f"  Extracted {n_samples} IMU samples")
        return True
    else:
        print(f"  Failed to extract IMU data")
        if stderr_path.exists():
            with open(stderr_path) as f:
                print(f"  Stderr: {f.read()[:500]}")
        return False


def get_video_info(video_path):
    """Get video resolution and duration."""
    with av.open(str(video_path)) as container:
        video = container.streams.video[0]
        width = video.width
        height = video.height
        duration = float(video.duration * video.time_base) if video.duration else 0
        fps = float(video.average_rate) if video.average_rate else 30
    return width, height, duration, fps


def downscale_video(input_path, output_path, target_width=2704, target_height=2028):
    """Downscale video to target resolution using FFmpeg.

    Returns True if successful, False otherwise.
    """
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)

    # Note: We don't copy data streams (GPMF/timecode) because:
    # 1. The timecode stream (tmcd) has "codec none" which FFmpeg can't write to MP4
    # 2. IMU extraction uses the original files anyway
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-i', str(input_path),
        '-vf', f'scale={target_width}:{target_height}',
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

        # Verify output file is valid
        if output_path.stat().st_size == 0:
            print(f"  Error: FFmpeg produced 0-byte file")
            return False

        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error downscaling: {e.stderr.decode()[:300] if e.stderr else 'Unknown error'}")
        return False


def analyze_trajectory(csv_path):
    """Analyze SLAM trajectory and report statistics."""
    if not csv_path.exists():
        return None

    import pandas as pd
    df = pd.read_csv(csv_path)

    total_frames = len(df)
    tracked_frames = (df['is_lost'] == 0).sum() if 'is_lost' in df.columns else total_frames
    tracking_rate = tracked_frames / total_frames * 100 if total_frames > 0 else 0

    # Calculate trajectory length
    if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
        positions = df[['x', 'y', 'z']].values
        valid_positions = positions[~np.isnan(positions).any(axis=1)]
        if len(valid_positions) > 1:
            diffs = np.diff(valid_positions, axis=0)
            trajectory_length = np.sum(np.linalg.norm(diffs, axis=1))
        else:
            trajectory_length = 0
    else:
        trajectory_length = 0

    return {
        'total_frames': total_frames,
        'tracked_frames': tracked_frames,
        'tracking_rate': tracking_rate,
        'trajectory_length': trajectory_length
    }


@click.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('-ct', '--camera_type', type=click.Choice(['gopro9', 'hero13']), default='gopro9',
              help='Camera type (affects mask and settings)')
@click.option('-o', '--output_dir', default=None, help='Output directory (default: temp dir)')
@click.option('-m', '--load_map', default=None, help='Load existing map for localization')
@click.option('-s', '--settings_file', default=None, help='Custom SLAM settings YAML')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="Skip docker pull")
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Don't use SLAM mask")
@click.option('-k', '--keep_output', is_flag=True, default=False, help="Keep output directory")
@click.option('-v', '--verbose', is_flag=True, default=False, help="Show SLAM output")
def main(video_path, camera_type, output_dir, load_map, settings_file,
         docker_image, no_docker_pull, no_mask, keep_output, verbose):
    """
    Test SLAM on a single video file.

    VIDEO_PATH: Path to the video file (MP4)
    """
    video_path = pathlib.Path(video_path).absolute()

    # Get video info
    width, height, duration, fps = get_video_info(video_path)
    print(f"Video: {video_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {duration:.1f}s @ {fps:.1f} fps")

    # Setup output directory
    if output_dir is None:
        output_dir = pathlib.Path(tempfile.mkdtemp(prefix='slam_test_'))
        print(f"Using temp directory: {output_dir}")
    else:
        output_dir = pathlib.Path(output_dir).absolute()
        output_dir.mkdir(parents=True, exist_ok=True)

    # Check resolution and downscale if needed
    expected_res = (2704, 2028) if camera_type == 'hero13' else (1920, 1080)
    needs_downscale = False

    if camera_type == 'hero13' and width >= 3000:
        # Hero 13 at 4K needs downscaling to 2.7K
        needs_downscale = True
        print(f"  Video is 4K, will downscale to {expected_res[0]}x{expected_res[1]}")
    elif (width, height) != expected_res:
        print(f"  Warning: Resolution {width}x{height} differs from expected {expected_res[0]}x{expected_res[1]}")

    # Prepare video for SLAM
    raw_video_path = output_dir / 'raw_video.mp4'
    if not raw_video_path.exists():
        if needs_downscale:
            print(f"Downscaling video to {output_dir}...")
            success = downscale_video(video_path, raw_video_path,
                                      target_width=expected_res[0],
                                      target_height=expected_res[1])
            if not success:
                print("  Failed to downscale video")
                if not keep_output and output_dir.name.startswith('slam_test_'):
                    shutil.rmtree(output_dir)
                return

            # Verify downscaled resolution
            new_w, new_h, _, _ = get_video_info(raw_video_path)
            print(f"  Downscaled to {new_w}x{new_h}")
        else:
            print(f"Copying video to {output_dir}...")
            shutil.copy(str(video_path), str(raw_video_path))

    # Extract IMU data
    # For IMU extraction, we need the original file (with GPMF metadata)
    # This is either the original video_path, or a *_4k_original file
    imu_path = output_dir / 'imu_data.json'
    if not imu_path.exists():
        print("Extracting IMU data...")
        # If we downscaled, the original is video_path itself
        # Otherwise, check for _4k_original backup
        original_4k = video_path.with_stem(video_path.stem + '_4k_original')
        if original_4k.exists():
            imu_source = original_4k
        else:
            imu_source = video_path
        success = extract_imu_data(imu_source, output_dir, no_docker_pull=no_docker_pull)
        if not success:
            print("  Make sure the video has GoPro GPMF metadata")
            if not keep_output and output_dir.name.startswith('slam_test_'):
                shutil.rmtree(output_dir)
            return

    # Select settings file
    if settings_file is None:
        if camera_type == 'hero13':
            settings_file = pathlib.Path(ROOT_DIR) / 'hero13_720p_slam_settings_gopro9_tbc.yaml'
        else:
            # Use default in Docker
            settings_file = None
    else:
        settings_file = pathlib.Path(settings_file).absolute()

    if settings_file and not settings_file.exists():
        print(f"Error: Settings file not found: {settings_file}")
        return

    print(f"SLAM settings: {settings_file.name if settings_file else 'Docker default (gopro10)'}")

    # Create SLAM mask
    if not no_mask:
        print("Creating SLAM mask...")
        if camera_type == 'hero13':
            slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
            slam_mask = draw_predefined_mask_hero13(
                slam_mask, color=255, mirror=True, finger=True)
        else:
            slam_mask = np.zeros((1080, 1920), dtype=np.uint8)
            slam_mask = draw_predefined_mask(
                slam_mask, color=255, mirror=True, gripper=False, finger=True)
        mask_path = output_dir / 'slam_mask.png'
        cv2.imwrite(str(mask_path), slam_mask)

    # Pull Docker image
    if not no_docker_pull:
        print(f"Pulling Docker image {docker_image}...")
        cmd = ['docker', 'pull', docker_image]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print("Docker pull failed!")
            print(result.stderr.decode())
            return

    # Build SLAM command
    mount_target = pathlib.Path('/data')
    csv_path = mount_target / 'camera_trajectory.csv'
    video_mount = mount_target / 'raw_video.mp4'
    json_mount = mount_target / 'imu_data.json'
    mask_mount = mount_target / 'slam_mask.png'

    cmd = [
        'docker', 'run', '--rm',
        '--volume', f'{output_dir}:/data',
    ]

    # Mount settings file if custom
    if settings_file:
        settings_mount = pathlib.Path('/settings') / settings_file.name
        cmd.extend(['--volume', f'{settings_file.parent}:/settings'])
        settings_arg = str(settings_mount)
    else:
        settings_arg = '/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml'

    # Mount map if loading
    if load_map:
        load_map = pathlib.Path(load_map).absolute()
        map_mount = pathlib.Path('/map') / load_map.name
        cmd.extend(['--volume', f'{load_map.parent}:/map'])

    cmd.extend([
        docker_image,
        '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
        '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
        '--setting', settings_arg,
        '--input_video', str(video_mount),
        '--input_imu_json', str(json_mount),
        '--output_trajectory_csv', str(csv_path),
    ])

    if load_map:
        cmd.extend(['--load_map', str(map_mount)])
    else:
        map_output = mount_target / 'map_atlas.osa'
        cmd.extend(['--save_map', str(map_output)])

    if not no_mask:
        cmd.extend(['--mask_img', str(mask_mount)])

    # Run SLAM
    print("\nRunning SLAM...")
    print(f"  Mode: {'Localization' if load_map else 'Mapping'}")

    stdout_path = output_dir / 'slam_stdout.txt'
    stderr_path = output_dir / 'slam_stderr.txt'

    timeout = max(duration * 16, 120)  # At least 2 minutes

    try:
        if verbose:
            result = subprocess.run(cmd, timeout=timeout)
        else:
            result = subprocess.run(
                cmd,
                stdout=stdout_path.open('w'),
                stderr=stderr_path.open('w'),
                timeout=timeout
            )
    except subprocess.TimeoutExpired:
        print(f"  SLAM timed out after {timeout:.0f}s")
        result = None

    # Analyze results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    trajectory_csv = output_dir / 'camera_trajectory.csv'
    if trajectory_csv.exists():
        stats = analyze_trajectory(trajectory_csv)
        if stats:
            print(f"  Total frames:     {stats['total_frames']}")
            print(f"  Tracked frames:   {stats['tracked_frames']}")
            print(f"  Tracking rate:    {stats['tracking_rate']:.1f}%")
            print(f"  Trajectory length: {stats['trajectory_length']:.3f}m")

            if stats['tracking_rate'] < 50:
                print("\n  ⚠️  Low tracking rate! Possible issues:")
                print("      - Poor lighting or motion blur in video")
                print("      - Incorrect camera settings/intrinsics")
                print("      - Missing or incorrect IMU data")
            elif stats['tracking_rate'] < 80:
                print("\n  ⚠️  Moderate tracking rate - may have gaps")
            else:
                print("\n  ✓ Good tracking rate!")
    else:
        print("  SLAM failed - no trajectory output")
        if stderr_path.exists():
            print("\n  Last 20 lines of stderr:")
            with open(stderr_path) as f:
                lines = f.readlines()
                for line in lines[-20:]:
                    print(f"    {line.rstrip()}")

    # Show output files
    print(f"\nOutput files in: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        if size > 1024*1024:
            size_str = f"{size/1024/1024:.1f}MB"
        elif size > 1024:
            size_str = f"{size/1024:.1f}KB"
        else:
            size_str = f"{size}B"
        print(f"  {f.name}: {size_str}")

    if not keep_output and output_dir.name.startswith('slam_test_'):
        print(f"\nCleaning up temp directory...")
        shutil.rmtree(output_dir)
    else:
        print(f"\nOutput kept at: {output_dir}")


if __name__ == '__main__':
    main()
