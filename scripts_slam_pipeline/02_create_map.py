"""
python scripts_slam_pipeline/00_process_videos.py -i data_workspace/toss_objects/20231113/mapping
"""

# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import multiprocessing
import concurrent.futures
import json
from tqdm import tqdm
import numpy as np
import cv2
import av
from umi.common.cv_util import draw_predefined_mask, draw_predefined_mask_hero13, draw_predefined_mask_grabette
from umi.common.camera_config import (
    CAMERA_CONFIGS,
    generate_slam_settings_for_resolution,
    generate_720p_slam_settings,
    downscale_video_for_slam,
    get_video_resolution
)

# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for mapping video')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-ct', '--camera_type', type=click.Choice(['gopro9', 'hero13', 'rpi_bno080', 'grabette']), default='gopro9',
              help='Camera type (gopro9 for Hero 9/10/11, hero13 for Hero 13, rpi_bno080/grabette for RPi camera)')
@click.option('-s', '--settings_file', default=None, help='SLAM settings YAML (auto-selected if not provided)')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Whether to mask out gripper and mirrors. Set if map is created with bare GoPro no on gripper.")
@click.option('--retries', type=int, default=0, help="Retry SLAM up to N times, keeping the best result (by tracking rate)")
@click.option('--quality_downscale', is_flag=True, default=False, help="Pre-downscale video to SLAM input resolution using ffmpeg (recommended for 4K input)")
@click.option('--slam_resolution', type=str, default=None, help="Override SLAM input resolution (e.g., '2704x2028'). Default uses camera config.")
def main(input_dir, map_path, camera_type, settings_file, docker_image, no_docker_pull, no_mask, retries, quality_downscale, slam_resolution):
    video_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    for fn in ['raw_video.mp4', 'imu_data.json']:
        assert video_dir.joinpath(fn).is_file()

    # Resample IMU to uniform 200Hz for RPi/grabette (ORB-SLAM3 requires uniform spacing)
    imu_filename = 'imu_data.json'
    if camera_type in ('rpi_bno080', 'grabette'):
        resampled_path = video_dir.joinpath('imu_data_resampled.json')
        if not resampled_path.is_file():
            print("Resampling IMU to uniform 200Hz...")
            from scripts.resample_imu import deduplicate_samples, resample_stream
            with open(video_dir.joinpath('imu_data.json')) as f:
                imu_raw = json.load(f)
            streams = imu_raw['1']['streams']
            for stream_name in ['ACCL', 'GYRO']:
                if stream_name not in streams:
                    continue
                samples = streams[stream_name]['samples']
                samples = deduplicate_samples(samples)
                resampled = resample_stream(samples, 200)
                streams[stream_name]['samples'] = resampled
                print(f"  {stream_name}: {len(samples)} deduped -> {len(resampled)} resampled")
            if 'ANGL' in streams:
                del streams['ANGL']
            with open(resampled_path, 'w') as f:
                json.dump({"1": {"streams": streams}}, f)
        imu_filename = 'imu_data_resampled.json'
        print(f"Using resampled IMU: {imu_filename}")

    # Get video resolution
    video_w, video_h = get_video_resolution(video_dir.joinpath('raw_video.mp4'))
    print(f"Video resolution: {video_w}x{video_h}")

    # Quality downscale: pre-process video to configured SLAM input resolution using ffmpeg
    slam_video_path = video_dir.joinpath('raw_video.mp4')
    config = CAMERA_CONFIGS.get(camera_type, CAMERA_CONFIGS['gopro9'])

    # Override slam_input_resolution if provided via parameter
    if slam_resolution:
        try:
            parts = slam_resolution.lower().split('x')
            slam_input_res = (int(parts[0]), int(parts[1]))
            print(f"Using custom SLAM resolution: {slam_input_res[0]}x{slam_input_res[1]}")
        except:
            raise ValueError(f"Invalid slam_resolution format: {slam_resolution}. Expected format: WIDTHxHEIGHT")
    else:
        slam_input_res = config.get('slam_input_resolution')

    if quality_downscale and slam_input_res is not None:
        target_w, target_h = slam_input_res
        if video_w > target_w or video_h > target_h:
            downscaled_path = video_dir.joinpath(f'raw_video_{target_w}x{target_h}.mp4')
            if not downscaled_path.exists():
                print(f"Pre-downscaling video to {target_w}x{target_h} using ffmpeg (high quality)...")
                success = downscale_video_for_slam(
                    video_dir.joinpath('raw_video.mp4'),
                    downscaled_path,
                    target_resolution=(target_w, target_h)
                )
                if not success:
                    print("Warning: Failed to downscale video, using original")
                else:
                    print(f"Created {downscaled_path}")
                    slam_video_path = downscaled_path
            else:
                print(f"Using existing downscaled video: {downscaled_path}")
                slam_video_path = downscaled_path

    # Get SLAM video resolution
    slam_video_w, slam_video_h = get_video_resolution(slam_video_path)

    # Determine settings file based on camera type
    if settings_file is None:
        if camera_type == 'hero13':
            # Auto-generate settings for the SLAM video resolution
            settings_path = video_dir.joinpath('slam_settings_auto.yaml')
            print(f"Auto-generating SLAM settings for {slam_video_w}x{slam_video_h}...")
            generate_slam_settings_for_resolution(
                camera_type='hero13',
                input_resolution=(slam_video_w, slam_video_h),
                output_path=settings_path
            )
        elif camera_type == 'rpi_bno080':
            # Use RPi+BNO080 settings - look for calibrated settings first, then fallback to template
            calibration_dir = pathlib.Path(ROOT_DIR) / 'example' / 'calibration'
            calibrated_settings = calibration_dir / 'rpi_bno080_calibrated_slam_settings.yaml'
            template_settings = pathlib.Path(ROOT_DIR) / 'rpi_bno080_slam_settings.yaml'
            if calibrated_settings.is_file():
                settings_path = calibrated_settings
                print(f"Using calibrated RPi+BNO080 settings: {settings_path}")
            elif template_settings.is_file():
                settings_path = template_settings
                print(f"WARNING: Using UNCALIBRATED template settings: {settings_path}")
                print("         Run calibration and update rpi_bno080_slam_settings.yaml for best results!")
            else:
                print("Error: No RPi+BNO080 settings file found")
                print(f"  Expected calibrated: {calibrated_settings}")
                print(f"  Or template: {template_settings}")
                exit(1)
        elif camera_type == 'grabette':
            # Use grabette (BMI088) settings
            template_settings = pathlib.Path(ROOT_DIR) / 'rpi_bmi088_slam_settings.yaml'
            if template_settings.is_file():
                settings_path = template_settings
                print(f"Using grabette (BMI088) settings: {settings_path}")
            else:
                print("Error: No grabette settings file found")
                print(f"  Expected: {template_settings}")
                exit(1)
        else:
            # Use built-in settings for GoPro 9/10/11 (inside docker)
            settings_path = None
    else:
        settings_path = pathlib.Path(os.path.expanduser(settings_file)).absolute()
        if not settings_path.is_file():
            print(f"Error: Settings file not found: {settings_path}")
            exit(1)

    if settings_path is not None:
        print(f"Using SLAM settings: {settings_path}")

    if map_path is None:
        map_path = video_dir.joinpath('map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    map_path.parent.mkdir(parents=True, exist_ok=True)

    # pull docker
    if not no_docker_pull:
        print(f"Pulling docker image {docker_image}")
        cmd = [
            'docker',
            'pull',
            docker_image
        ]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            print("Docker pull failed!")
            exit(1)

    mount_target = pathlib.Path('/data')
    csv_path = mount_target.joinpath('mapping_camera_trajectory.csv')
    # Use downscaled video if available
    video_path = mount_target.joinpath(slam_video_path.name)
    json_path = mount_target.joinpath(imu_filename)
    mask_path = mount_target.joinpath('slam_mask.png')
    if not no_mask:
        mask_write_path = video_dir.joinpath('slam_mask.png')
        # Get video resolution for mask (should match SLAM video, not original)
        with av.open(str(slam_video_path)) as container:
            stream = container.streams.video[0]
            mask_h, mask_w = stream.height, stream.width
        slam_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        # Select mask function based on camera type
        if camera_type == 'hero13':
            slam_mask = draw_predefined_mask_hero13(
                slam_mask, color=255, mirror=True, finger=True)
        elif camera_type in ('rpi_bno080', 'grabette'):
            slam_mask = draw_predefined_mask_grabette(
                slam_mask, color=255, device=True)
        else:
            slam_mask = draw_predefined_mask(
                slam_mask, color=255, mirror=True, gripper=False, finger=True)
        cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

    map_mount_source = pathlib.Path(map_path)
    map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)

    # Determine settings argument for docker
    if settings_path is not None:
        settings_mount_target = pathlib.Path('/settings').joinpath(settings_path.name)
        settings_arg = str(settings_mount_target)
    else:
        # Use built-in settings for GoPro 9/10/11
        settings_arg = '/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml'

    # run SLAM
    cmd = [
        'docker',
        'run',
        '--rm', # delete after finish
        '--volume', str(video_dir) + ':' + '/data',
        '--volume', str(map_mount_source.parent) + ':' + str(map_mount_target.parent),
    ]
    # Mount custom settings file if provided
    if settings_path is not None:
        cmd.extend(['--volume', str(settings_path.parent) + ':' + '/settings'])
    cmd.extend([
        docker_image,
        '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
        '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
        '--setting', settings_arg,
        '--input_video', str(video_path),
        '--input_imu_json', str(json_path),
        '--output_trajectory_csv', str(csv_path),
        '--save_map', str(map_mount_target)
    ])
    if not no_mask:
        cmd.extend([
            '--mask_img', str(mask_path)
        ])

    import pandas as pd

    def copy_file(src, dst):
        """Copy file handling root-owned Docker files."""
        if src.is_file():
            data = src.read_bytes()
            if dst.is_file():
                os.remove(str(dst))
            dst.write_bytes(data)

    total_attempts = 1 + retries
    best_pct = -1
    best_attempt = 0

    for attempt in range(1, total_attempts + 1):
        if total_attempts > 1:
            print(f"\n--- Attempt {attempt}/{total_attempts} ---")
        else:
            print("Running SLAM mapping...")

        stdout_path = video_dir.joinpath('slam_stdout.txt')
        stderr_path = video_dir.joinpath('slam_stderr.txt')

        result = subprocess.run(
            cmd,
            cwd=str(video_dir),
            stdout=stdout_path.open('w'),
            stderr=stderr_path.open('w')
        )

        traj_path = video_dir.joinpath('mapping_camera_trajectory.csv')
        if result.returncode != 0 or not traj_path.is_file():
            print(f"  SLAM failed (return code {result.returncode})")
            continue

        df = pd.read_csv(traj_path)
        tracked = len(df) - df['is_lost'].sum()
        pct = 100 * tracked / len(df) if len(df) > 0 else 0
        print(f"  Tracking: {tracked}/{len(df)} ({pct:.1f}%)")

        if pct > best_pct:
            best_pct = pct
            best_attempt = attempt
            # Save best result
            if total_attempts > 1:
                for src, dst_name in [
                    (traj_path, 'mapping_camera_trajectory_best.csv'),
                    (map_path, 'map_atlas_best.osa'),
                    (stdout_path, 'slam_stdout_best.txt'),
                ]:
                    copy_file(src, video_dir / dst_name)

        if pct >= 90:
            if total_attempts > 1:
                print(f"  >= 90% tracking, stopping early")
            break

    # Restore best result if we did retries and last attempt wasn't the best
    if total_attempts > 1 and best_pct >= 0 and best_attempt != attempt:
        copy_file(video_dir / 'mapping_camera_trajectory_best.csv', traj_path)
        copy_file(video_dir / 'map_atlas_best.osa', map_path)

    if total_attempts > 1:
        print(f"\nBest result: attempt {best_attempt}/{total_attempts} ({best_pct:.1f}% tracking)")

    # Two-pass: re-localize against the map to recover init frames
    # Even a partial map (>0%) can help — pass 2 may track frames that pass 1 missed
    if best_pct > 0 and map_path.is_file():
        print("\nRunning pass 2 (re-localization to recover init frames)...")
        csv_path_pass2 = mount_target.joinpath('mapping_camera_trajectory_pass2.csv')
        cmd_pass2 = [
            'docker', 'run', '--rm',
            '--volume', str(video_dir) + ':' + '/data',
            '--volume', str(map_mount_source.parent) + ':' + str(map_mount_target.parent),
        ]
        if settings_path is not None:
            cmd_pass2.extend(['--volume', str(settings_path.parent) + ':' + '/settings'])
        cmd_pass2.extend([
            docker_image,
            '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
            '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
            '--setting', settings_arg,
            '--input_video', str(video_path),
            '--input_imu_json', str(json_path),
            '--output_trajectory_csv', str(csv_path_pass2),
            '--load_map', str(map_mount_target),
        ])
        if not no_mask:
            cmd_pass2.extend(['--mask_img', str(mask_path)])

        stdout_path_pass2 = video_dir.joinpath('slam_stdout_pass2.txt')
        stderr_path_pass2 = video_dir.joinpath('slam_stderr_pass2.txt')

        # Timeout: pass 1 processes at ~50 FPS, give pass 2 generous 5x margin
        with av.open(str(slam_video_path)) as container:
            video_duration = float(container.streams.video[0].duration * container.streams.video[0].time_base)
        pass2_timeout = max(video_duration * 10, 120)

        try:
            result_pass2 = subprocess.run(
                cmd_pass2,
                cwd=str(video_dir),
                stdout=stdout_path_pass2.open('w'),
                stderr=stderr_path_pass2.open('w'),
                timeout=pass2_timeout
            )
        except subprocess.TimeoutExpired:
            print(f"  Pass 2 timed out after {pass2_timeout:.0f}s, keeping pass 1")
            result_pass2 = None

        traj_pass2 = video_dir.joinpath('mapping_camera_trajectory_pass2.csv')
        if result_pass2 is not None and result_pass2.returncode == 0 and traj_pass2.is_file():
            df2 = pd.read_csv(traj_pass2)
            tracked2 = len(df2) - df2['is_lost'].sum()
            pct2 = 100 * tracked2 / len(df2) if len(df2) > 0 else 0
            print(f"  Pass 2: {tracked2}/{len(df2)} ({pct2:.1f}%)")

            if pct2 > best_pct:
                # Pass 2 is better — use it as the final result
                copy_file(traj_pass2, video_dir / 'mapping_camera_trajectory.csv')
                print(f"  Pass 2 improved tracking: {best_pct:.1f}% -> {pct2:.1f}%")
            else:
                print(f"  Pass 2 did not improve ({pct2:.1f}% vs {best_pct:.1f}%), keeping pass 1")
        else:
            print(f"  Pass 2 failed (return code {result_pass2.returncode}), keeping pass 1")


# %%
if __name__ == "__main__":
    main()
