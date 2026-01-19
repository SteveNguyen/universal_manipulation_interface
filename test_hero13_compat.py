#!/usr/bin/env python3
"""
Quick test script to check GoPro Hero 13 compatibility with UMI pipeline.
Usage: python test_hero13_compat.py /path/to/hero13_video.mp4
"""

import sys
import pathlib
import json
import subprocess
import av
from exiftool import ExifToolHelper

def test_video_metadata(video_path):
    """Test if video has required metadata."""
    print("=" * 60)
    print("TEST 1: Video Metadata")
    print("=" * 60)

    try:
        with ExifToolHelper() as et:
            meta = list(et.get_metadata(str(video_path)))[0]

            # Check for camera serial
            if 'QuickTime:CameraSerialNumber' in meta:
                print(f"✓ Camera Serial Number: {meta['QuickTime:CameraSerialNumber']}")
            else:
                print("✗ Camera Serial Number: NOT FOUND")
                print("  Available QuickTime fields:")
                for key in sorted(meta.keys()):
                    if key.startswith('QuickTime:'):
                        print(f"    - {key}")
                return False

            # Check video resolution
            if 'File:ImageWidth' in meta and 'File:ImageHeight' in meta:
                width = meta['File:ImageWidth']
                height = meta['File:ImageHeight']
                print(f"✓ Resolution: {width} x {height}")

            # Check frame rate
            if 'QuickTime:VideoFrameRate' in meta:
                fps = meta['QuickTime:VideoFrameRate']
                print(f"✓ Frame Rate: {fps} fps")

            return True
    except Exception as e:
        print(f"✗ Error reading metadata: {e}")
        return False


def test_video_timing(video_path):
    """Test if video has timecode metadata for synchronization."""
    print("\n" + "=" * 60)
    print("TEST 2: Video Timing/Timecode")
    print("=" * 60)

    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]

            # Check for timecode
            if 'timecode' in stream.metadata:
                print(f"✓ Timecode: {stream.metadata['timecode']}")
            else:
                print("✗ Timecode: NOT FOUND")
                print(f"  Available metadata keys: {list(stream.metadata.keys())}")
                return False

            # Check for creation time
            if 'creation_time' in stream.metadata:
                print(f"✓ Creation Time: {stream.metadata['creation_time']}")
            else:
                print("✗ Creation Time: NOT FOUND")
                return False

            # Check frame rate
            fps = stream.average_rate
            print(f"✓ Average Frame Rate: {fps}")

            return True
    except Exception as e:
        print(f"✗ Error reading video stream: {e}")
        return False


def test_imu_extraction(video_path):
    """Test if IMU data can be extracted (requires Docker)."""
    print("\n" + "=" * 60)
    print("TEST 3: IMU Data Extraction")
    print("=" * 60)

    video_path = pathlib.Path(video_path)
    video_dir = video_path.parent

    # Create test directory
    test_dir = video_dir / "imu_test"
    test_dir.mkdir(exist_ok=True)

    test_video = test_dir / "raw_video.mp4"

    # Remove existing file/symlink if present
    if test_video.exists() or test_video.is_symlink():
        test_video.unlink()

    # Copy the video file instead of symlinking (Docker needs actual file)
    print(f"Copying video to test directory (this may take a moment)...")
    import shutil
    shutil.copy2(video_path, test_video)
    print(f"✓ Video copied: {test_video.stat().st_size / 1024 / 1024:.1f} MB")

    json_path = test_dir / "imu_data.json"
    if json_path.exists():
        json_path.unlink()

    # Check if docker is available
    try:
        result = subprocess.run(['docker', '--version'],
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("✗ Docker not available")
            print("  Install Docker to test IMU extraction")
            return None
    except FileNotFoundError:
        print("✗ Docker not installed")
        print("  Install Docker to test IMU extraction")
        return None

    print("Pulling OpenImuCameraCalibrator Docker image...")
    result = subprocess.run([
        'docker', 'pull', 'chicheng/openicc:latest'
    ], capture_output=True)

    if result.returncode != 0:
        print("✗ Failed to pull Docker image")
        return False

    print("Extracting IMU data (this may take a moment)...")

    # Run IMU extraction
    cmd = [
        'docker', 'run', '--rm',
        '--volume', f'{test_dir.absolute()}:/data',
        'chicheng/openicc:latest',
        'node',
        '/OpenImuCameraCalibrator/javascript/extract_metadata_single.js',
        '/data/raw_video.mp4',
        '/data/imu_data.json'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if json_path.exists():
        # Parse and check IMU data
        try:
            with open(json_path) as f:
                imu_data = json.load(f)

            # Check structure
            if '1' in imu_data and 'streams' in imu_data['1']:
                streams = imu_data['1']['streams']

                if 'ACCL' in streams:
                    accl_samples = len(streams['ACCL']['samples'])
                    print(f"✓ Accelerometer samples: {accl_samples}")
                else:
                    print("✗ No accelerometer data")
                    return False

                if 'GYRO' in streams:
                    gyro_samples = len(streams['GYRO']['samples'])
                    print(f"✓ Gyroscope samples: {gyro_samples}")
                else:
                    print("✗ No gyroscope data")
                    return False

                # Check sample format
                sample = streams['ACCL']['samples'][0]
                if 'value' in sample and 'cts' in sample:
                    print(f"✓ Sample format correct")
                    print(f"  First ACCL value: {sample['value']}")
                    print(f"  Timestamp: {sample['cts']} ms")
                else:
                    print("✗ Sample format incorrect")
                    return False

                return True
            else:
                print("✗ IMU data structure incorrect")
                print(f"  Keys found: {list(imu_data.keys())}")
                return False

        except Exception as e:
            print(f"✗ Error parsing IMU data: {e}")
            return False
    else:
        print("✗ IMU extraction failed - no output file created")
        print(f"  stderr: {result.stderr}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_hero13_compat.py /path/to/hero13_video.mp4")
        sys.exit(1)

    video_path = pathlib.Path(sys.argv[1])

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"Testing GoPro Hero 13 Compatibility")
    print(f"Video: {video_path.name}")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Metadata", test_video_metadata(video_path)))
    results.append(("Timing", test_video_timing(video_path)))
    results.append(("IMU", test_imu_extraction(video_path)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, result in results:
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
            all_passed = False
        else:
            status = "⊘ SKIP"
        print(f"{name:20} {status}")

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ Your GoPro Hero 13 is compatible!")
        print("\nNext steps:")
        print("1. Calibrate your camera using GOPRO_HERO13_CALIBRATION.md")
        print("2. Create intrinsics JSON file")
        print("3. Update pipeline scripts to use your intrinsics")
        print("4. Test with example_demo_session workflow")
    else:
        print("✗ Compatibility issues found")
        print("\nTroubleshooting:")
        print("- Ensure GoPro Labs firmware is installed")
        print("- Check that video was recorded with Labs features enabled")
        print("- Verify camera settings (date/time, no hypersmooth)")
    print("=" * 60)


if __name__ == "__main__":
    main()
