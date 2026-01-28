"""
python scripts_slam_pipeline/03_batch_slam.py -i data_workspace/fold_cloth_20231214/demos
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
import cv2
import av
import numpy as np
from umi.common.cv_util import draw_predefined_mask, draw_predefined_mask_hero13
from umi.common.camera_config import (
    CAMERA_CONFIGS,
    generate_slam_settings_for_resolution,
    generate_720p_slam_settings,
    downscale_video_for_slam,
    get_video_resolution
)


# %%
def runner(cmd, cwd, stdout_path, stderr_path, timeout, **kwargs):
    try:
        return subprocess.run(cmd,                       
            cwd=str(cwd),
            stdout=stdout_path.open('w'),
            stderr=stderr_path.open('w'),
            timeout=timeout,
            **kwargs)
    except subprocess.TimeoutExpired as e:
        return e


# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for demos folder')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-ct', '--camera_type', type=click.Choice(['gopro9', 'hero13']), default='gopro9',
              help='Camera type (gopro9 for Hero 9/10/11, hero13 for Hero 13)')
@click.option('-s', '--settings_file', default=None, help='SLAM settings YAML (auto-selected if not provided)')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-n', '--num_workers', type=int, default=None)
@click.option('-ml', '--max_lost_frames', type=int, default=60)
@click.option('-tm', '--timeout_multiple', type=float, default=16, help='timeout_multiple * duration = timeout')
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
@click.option('--quality_downscale', is_flag=True, default=False, help="Pre-downscale videos to SLAM input resolution using ffmpeg (recommended for 4K input)")
def main(input_dir, map_path, camera_type, settings_file, docker_image, num_workers, max_lost_frames, timeout_multiple, no_docker_pull, quality_downscale):
    input_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    input_video_dirs = [x.parent for x in input_dir.glob('demo*/raw_video.mp4')]
    input_video_dirs += [x.parent for x in input_dir.glob('map*/raw_video.mp4')]
    print(f'Found {len(input_video_dirs)} video dirs')

    # Get camera config for determining SLAM input resolution
    config = CAMERA_CONFIGS.get(camera_type, CAMERA_CONFIGS['gopro9'])
    slam_input_res = config.get('slam_input_resolution')

    # Determine SLAM video resolution (after potential quality downscale)
    slam_video_w, slam_video_h = None, None
    if camera_type == 'hero13' and input_video_dirs:
        sample_video = input_video_dirs[0].joinpath('raw_video.mp4')
        video_w, video_h = get_video_resolution(sample_video)
        print(f"Video resolution: {video_w}x{video_h}")

        if quality_downscale and slam_input_res is not None:
            target_w, target_h = slam_input_res
            if video_w > target_w or video_h > target_h:
                print(f"Will downscale to {target_w}x{target_h} for SLAM")
                slam_video_w, slam_video_h = target_w, target_h
            else:
                slam_video_w, slam_video_h = video_w, video_h
        else:
            slam_video_w, slam_video_h = video_w, video_h

    # Determine settings file based on camera type
    if settings_file is None:
        if camera_type == 'hero13':
            if slam_video_w is not None:
                # Generate settings for SLAM video resolution
                settings_path = input_dir.joinpath('slam_settings_auto.yaml')
                print(f"Auto-generating SLAM settings for {slam_video_w}x{slam_video_h}...")
                generate_slam_settings_for_resolution(
                    camera_type='hero13',
                    input_resolution=(slam_video_w, slam_video_h),
                    output_path=settings_path
                )
            else:
                print("Error: No videos found to determine resolution")
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
        settings_mount_target = pathlib.Path('/settings').joinpath(settings_path.name)
        settings_arg = str(settings_mount_target)
    else:
        # Use built-in settings for GoPro 9/10/11
        settings_arg = '/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml'

    if map_path is None:
        map_path = input_dir.joinpath('mapping', 'map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    assert map_path.is_file()

    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 2

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

    with tqdm(total=len(input_video_dirs)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for video_dir in tqdm(input_video_dirs):
                video_dir = video_dir.absolute()
                if video_dir.joinpath('camera_trajectory.csv').is_file():
                    print(f"camera_trajectory.csv already exists, skipping {video_dir.name}")
                    continue
                
                # softlink won't work in bind volume
                mount_target = pathlib.Path('/data')
                csv_path = mount_target.joinpath('camera_trajectory.csv')
                json_path = mount_target.joinpath('imu_data.json')
                mask_path = mount_target.joinpath('slam_mask.png')
                mask_write_path = video_dir.joinpath('slam_mask.png')

                # find video duration and resolution
                with av.open(str(video_dir.joinpath('raw_video.mp4').absolute())) as container:
                    video = container.streams.video[0]
                    duration_sec = float(video.duration * video.time_base)
                    video_h, video_w = video.height, video.width
                timeout = duration_sec * timeout_multiple

                # Quality downscale: pre-process video to SLAM input resolution using ffmpeg
                slam_video_path = video_dir.joinpath('raw_video.mp4')
                if quality_downscale and slam_input_res is not None:
                    target_w, target_h = slam_input_res
                    if video_w > target_w or video_h > target_h:
                        downscaled_path = video_dir.joinpath(f'raw_video_{target_w}x{target_h}.mp4')
                        if not downscaled_path.exists():
                            success = downscale_video_for_slam(
                                video_dir.joinpath('raw_video.mp4'),
                                downscaled_path,
                                target_resolution=(target_w, target_h)
                            )
                            if not success:
                                print(f"Warning: Failed to downscale {video_dir.name}, using original")
                            else:
                                slam_video_path = downscaled_path
                        else:
                            slam_video_path = downscaled_path

                video_path = mount_target.joinpath(slam_video_path.name)

                # Get SLAM video resolution for mask
                with av.open(str(slam_video_path.absolute())) as container:
                    slam_video = container.streams.video[0]
                    mask_h, mask_w = slam_video.height, slam_video.width

                slam_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
                # Select mask function based on camera type
                if camera_type == 'hero13':
                    slam_mask = draw_predefined_mask_hero13(
                        slam_mask, color=255, mirror=True, finger=True)
                else:
                    slam_mask = draw_predefined_mask(
                        slam_mask, color=255, mirror=True, gripper=False, finger=True)
                cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

                map_mount_source = map_path
                map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)

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
                    '--load_map', str(map_mount_target),
                    '--mask_img', str(mask_path),
                    '--max_lost_frames', str(max_lost_frames)
                ])

                stdout_path = video_dir.joinpath('slam_stdout.txt')
                stderr_path = video_dir.joinpath('slam_stderr.txt')

                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(runner,
                    cmd, str(video_dir), stdout_path, stderr_path, timeout))
                # print(' '.join(cmd))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print("Done! Result:")
    print([x.result() for x in completed])

# %%
if __name__ == "__main__":
    main()
