"""
Main script for UMI SLAM pipeline.
python run_slam_pipeline.py <session_dir>
"""

import sys
import os

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import pandas as pd


def analyze_trajectory(csv_path):
    """Analyze tracking quality from trajectory CSV."""
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    total = len(df)
    lost = df['is_lost'].sum()
    tracked = total - lost
    pct = 100 * tracked / total if total > 0 else 0
    return {'total': total, 'tracked': tracked, 'lost': lost, 'pct': pct}


def print_mapping_summary(mapping_dir):
    """Print mapping tracking quality summary."""
    print("\n" + "="*60)
    print("MAPPING RESULTS")
    print("="*60)

    # Check for pass 2 trajectory (preferred)
    traj_pass2 = mapping_dir / 'mapping_camera_trajectory_pass2.csv'
    traj_pass1 = mapping_dir / 'mapping_camera_trajectory.csv'

    if traj_pass2.is_file():
        stats = analyze_trajectory(traj_pass2)
        if stats:
            print(f"Pass 2 (final): {stats['tracked']}/{stats['total']} frames tracked ({stats['pct']:.1f}%)")

    if traj_pass1.is_file():
        stats = analyze_trajectory(traj_pass1)
        if stats:
            print(f"Pass 1: {stats['tracked']}/{stats['total']} frames tracked ({stats['pct']:.1f}%)")

    map_path = mapping_dir / 'map_atlas.osa'
    if map_path.is_file():
        size_mb = map_path.stat().st_size / (1024 * 1024)
        print(f"Map size: {size_mb:.1f} MB")
    print("="*60 + "\n")


def print_batch_slam_summary(demo_dir):
    """Print batch SLAM tracking quality summary."""
    print("\n" + "="*60)
    print("BATCH SLAM RESULTS")
    print("="*60)

    demo_dirs = list(demo_dir.glob('demo_*'))
    results = []

    for d in demo_dirs:
        traj_file = d / 'camera_trajectory.csv'
        if traj_file.exists():
            stats = analyze_trajectory(traj_file)
            if stats:
                results.append(stats)

    total_demos = len(demo_dirs)
    success_demos = len(results)

    print(f"Demos processed: {success_demos}/{total_demos} ({100*success_demos/total_demos:.1f}%)")

    if results:
        excellent = sum(1 for r in results if r['pct'] >= 90)
        good = sum(1 for r in results if 80 <= r['pct'] < 90)
        medium = sum(1 for r in results if 50 <= r['pct'] < 80)
        poor = sum(1 for r in results if r['pct'] < 50)

        print(f"\nTracking quality:")
        print(f"  Excellent (>=90%): {excellent}/{success_demos} ({100*excellent/success_demos:.1f}%)")
        print(f"  Good (80-90%):     {good}/{success_demos} ({100*good/success_demos:.1f}%)")
        print(f"  Medium (50-80%):   {medium}/{success_demos} ({100*medium/success_demos:.1f}%)")
        print(f"  Poor (<50%):       {poor}/{success_demos} ({100*poor/success_demos:.1f}%)")

        avg_pct = sum(r['pct'] for r in results) / len(results)
        print(f"\nAverage tracking quality: {avg_pct:.1f}%")

    print("="*60 + "\n")


def print_final_summary(session):
    """Print final pipeline summary."""
    demo_dir = session / 'demos'
    demo_dirs = list(demo_dir.glob('demo_*'))

    # Count demos with trajectories
    with_traj = sum(1 for d in demo_dirs if (d / 'camera_trajectory.csv').exists())

    # Count demos with ArUco detections
    with_aruco = sum(1 for d in demo_dirs if (d / 'tag_detection.pkl').exists())

    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Dataset: {session.name}")
    print(f"Total demos: {len(demo_dirs)}")
    print(f"Demos with SLAM tracking: {with_traj}/{len(demo_dirs)} ({100*with_traj/len(demo_dirs):.1f}%)")
    print(f"Demos with ArUco detection: {with_aruco}/{len(demo_dirs)} ({100*with_aruco/len(demo_dirs):.1f}%)")

    dataset_plan = session / 'dataset_plan.pkl'
    if dataset_plan.is_file():
        print(f"\n✓ Dataset plan generated: {dataset_plan}")
        print("  Ready for training!")

    print("="*60 + "\n")


# %%
@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-c', '--calibration_dir', type=str, default=None)
@click.option('-ct', '--camera_type', type=click.Choice(['gopro9', 'hero13']),
              default='gopro9', help='Camera type (gopro9 for Hero 9/10/11, hero13 for Hero 13)')
def main(session_dir, calibration_dir, camera_type):
    script_dir = pathlib.Path(__file__).parent.joinpath('scripts_slam_pipeline')
    if calibration_dir is None:
        calibration_dir = pathlib.Path(__file__).parent.joinpath('example', 'calibration')
    else:
        calibration_dir = pathlib.Path(calibration_dir)
    assert calibration_dir.is_dir()

    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()

        print("############## 00_process_videos #############")
        script_path = script_dir.joinpath("00_process_videos.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 01_extract_gopro_imu ###########")
        script_path = script_dir.joinpath("01_extract_gopro_imu.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 02_create_map ###########")
        demo_dir = session.joinpath('demos')
        mapping_dir = demo_dir.joinpath('mapping')
        assert mapping_dir.is_dir()
        map_path = mapping_dir.joinpath('map_atlas.osa')

        script_path = script_dir.joinpath("02_create_map.py")
        assert script_path.is_file()

        if not map_path.is_file():
            cmd = [
                'python', str(script_path),
                '--input_dir', str(mapping_dir),
                '--map_path', str(map_path),
                '--camera_type', camera_type
            ]
            result = subprocess.run(cmd)
            assert result.returncode == 0
            assert map_path.is_file()
            print_mapping_summary(mapping_dir)

        print("############# 03_batch_slam ###########")
        script_path = script_dir.joinpath("03_batch_slam.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input_dir', str(demo_dir),
            '--map_path', str(map_path),
            '--camera_type', camera_type
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0
        print_batch_slam_summary(demo_dir)

        print("############# 04_detect_aruco ###########")
        script_path = script_dir.joinpath("04_detect_aruco.py")
        assert script_path.is_file()
        if camera_type == 'hero13':
            # Use 4K intrinsics for ArUco detection on native 4K videos
            camera_intrinsics = calibration_dir.joinpath('hero13_proper_intrinsics_4k.json')
        else:
            camera_intrinsics = calibration_dir.joinpath('gopro_intrinsics_2_7k.json')
        aruco_config = calibration_dir.joinpath('aruco_config.yaml')
        assert camera_intrinsics.is_file(), f"Camera intrinsics not found: {camera_intrinsics}"
        assert aruco_config.is_file()

        cmd = [
            'python', str(script_path),
            '--input_dir', str(demo_dir),
            '--camera_intrinsics', str(camera_intrinsics),
            '--aruco_yaml', str(aruco_config)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 05_run_calibrations ###########")
        script_path = script_dir.joinpath("05_run_calibrations.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 06_generate_dataset_plan ###########")
        script_path = script_dir.joinpath("06_generate_dataset_plan.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input', str(session),
            '--camera_type', camera_type
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        # Print final summary
        print_final_summary(session)

## %%
if __name__ == "__main__":
    main()
