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
from tqdm import tqdm
import numpy as np
import cv2
import av
from umi.common.cv_util import draw_predefined_mask, draw_predefined_mask_hero13
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
@click.option('-ct', '--camera_type', type=click.Choice(['gopro9', 'hero13']), default='gopro9',
              help='Camera type (gopro9 for Hero 9/10/11, hero13 for Hero 13)')
@click.option('-s', '--settings_file', default=None, help='SLAM settings YAML (auto-selected if not provided)')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Whether to mask out gripper and mirrors. Set if map is created with bare GoPro no on gripper.")
@click.option('--two_pass', is_flag=True, default=False, help="Run two-pass mapping: first pass creates map, second pass re-processes to capture initially missed frames")
@click.option('--quality_downscale', is_flag=True, default=False, help="Pre-downscale video to SLAM input resolution using ffmpeg (recommended for 4K input)")
def main(input_dir, map_path, camera_type, settings_file, docker_image, no_docker_pull, no_mask, two_pass, quality_downscale):
    video_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    for fn in ['raw_video.mp4', 'imu_data.json']:
        assert video_dir.joinpath(fn).is_file()

    # Get video resolution
    video_w, video_h = get_video_resolution(video_dir.joinpath('raw_video.mp4'))
    print(f"Video resolution: {video_w}x{video_h}")

    # Quality downscale: pre-process video to configured SLAM input resolution using ffmpeg
    slam_video_path = video_dir.joinpath('raw_video.mp4')
    config = CAMERA_CONFIGS.get(camera_type, CAMERA_CONFIGS['gopro9'])
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
    json_path = mount_target.joinpath('imu_data.json')
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

    stdout_path = video_dir.joinpath('slam_stdout.txt')
    stderr_path = video_dir.joinpath('slam_stderr.txt')

    print("Running SLAM mapping (pass 1)...")
    result = subprocess.run(
        cmd,
        cwd=str(video_dir),
        stdout=stdout_path.open('w'),
        stderr=stderr_path.open('w')
    )
    print(f"Pass 1 result: {result}")

    # Two-pass mapping: re-process video with existing map to capture initially missed frames
    # This helps when SLAM takes time to initialize and misses the first N frames
    if two_pass and map_path.is_file():
        print("\nRunning two-pass mapping (pass 2) to capture missed initial frames...")

        # Second pass: load existing map and re-process same video
        csv_path_pass2 = mount_target.joinpath('mapping_camera_trajectory_pass2.csv')
        cmd_pass2 = [
            'docker',
            'run',
            '--rm',
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
            '--load_map', str(map_mount_target),  # Load existing map
            '--save_map', str(map_mount_target),  # Save updated map
        ])
        if not no_mask:
            cmd_pass2.extend(['--mask_img', str(mask_path)])

        stdout_path_pass2 = video_dir.joinpath('slam_stdout_pass2.txt')
        stderr_path_pass2 = video_dir.joinpath('slam_stderr_pass2.txt')

        result_pass2 = subprocess.run(
            cmd_pass2,
            cwd=str(video_dir),
            stdout=stdout_path_pass2.open('w'),
            stderr=stderr_path_pass2.open('w')
        )
        print(f"Pass 2 result: {result_pass2}")

        # Analyze improvement
        traj_pass1 = video_dir.joinpath('mapping_camera_trajectory.csv')
        traj_pass2 = video_dir.joinpath('mapping_camera_trajectory_pass2.csv')
        if traj_pass1.is_file() and traj_pass2.is_file():
            import pandas as pd
            df1 = pd.read_csv(traj_pass1)
            df2 = pd.read_csv(traj_pass2)
            lost1 = df1['is_lost'].sum()
            lost2 = df2['is_lost'].sum()
            print(f"\nPass 1: {len(df1) - lost1}/{len(df1)} frames tracked ({100*(len(df1)-lost1)/len(df1):.1f}%)")
            print(f"Pass 2: {len(df2) - lost2}/{len(df2)} frames tracked ({100*(len(df2)-lost2)/len(df2):.1f}%)")


# %%
if __name__ == "__main__":
    main()
