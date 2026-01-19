#!/usr/bin/env python3
"""
Quick fisheye camera calibration for GoPro Hero 13.
Usage: python calibrate_hero13.py --video calib_video.mp4 --pattern checkerboard
"""

import cv2
import numpy as np
import json
import click
import sys
import multiprocessing
from functools import partial
from pathlib import Path


def process_frame_checkerboard(args):
    """Process a single frame for checkerboard detection (worker function)."""
    frame_data, frame_num, cols, rows, debug_path = args

    gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(
        gray, (cols, rows),
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        # Refine corner locations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        debug_img = None
        if debug_path is not None:
            debug_img = frame_data.copy()
            cv2.drawChessboardCorners(debug_img, (cols, rows), corners2, ret)

        return (True, corners2, debug_img, frame_num)

    return (False, None, None, frame_num)


def process_frame_charuco(args):
    """Process a single frame for ChArUco detection (worker function)."""
    frame_data, frame_num, cols, rows, square_size, marker_size, debug_path = args

    gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)

    # Create ArUco detector for this worker
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    charuco_board = cv2.aruco.CharucoBoard(
        (cols, rows),
        square_size,
        marker_size,
        aruco_dict
    )
    charuco_detector = cv2.aruco.CharucoDetector(charuco_board)

    # Detect ChArUco board
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    if charuco_corners is not None and len(charuco_corners) > 10:
        # Get object points for detected corners
        objp_charuco = charuco_board.getChessboardCorners()
        valid_ids = charuco_ids.flatten()
        obj_pts = objp_charuco[valid_ids].reshape(-1, 1, 3).astype(np.float64)
        img_pts = charuco_corners.astype(np.float64)

        debug_img = None
        if debug_path is not None:
            debug_img = frame_data.copy()
            if marker_corners is not None and marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(debug_img, marker_corners, marker_ids)
            cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_ids)

        return (True, obj_pts, img_pts, len(charuco_corners), debug_img, frame_num)

    return (False, None, None, 0, None, frame_num)


@click.command()
@click.option('--video', required=True, help='Path to calibration video')
@click.option('--pattern', type=click.Choice(['checkerboard', 'charuco']),
              default='checkerboard', help='Calibration pattern type')
@click.option('--rows', type=int, default=6,
              help='Number of inner corners (rows) for checkerboard, or ChArUco board rows')
@click.option('--cols', type=int, default=9,
              help='Number of inner corners (cols) for checkerboard, or ChArUco board cols')
@click.option('--square_size', type=float, default=0.025,
              help='Square size in meters (e.g., 0.025 for 25mm)')
@click.option('--marker_size', type=float, default=0.019,
              help='ArUco marker size in meters (for ChArUco only)')
@click.option('--frame_skip', type=int, default=10,
              help='Process every Nth frame to avoid similar views')
@click.option('--min_frames', type=int, default=20,
              help='Minimum frames needed for calibration')
@click.option('--output', default='hero13_intrinsics.json',
              help='Output JSON file path')
@click.option('--debug_dir', default='calibration_debug',
              help='Directory to save debug images with detected corners')
@click.option('--no_debug', is_flag=True,
              help='Skip saving debug images')
@click.option('--workers', type=int, default=None,
              help='Number of parallel workers (default: CPU count)')
def main(video, pattern, rows, cols, square_size, marker_size, frame_skip, min_frames, output, debug_dir, no_debug, workers):
    """Calibrate GoPro Hero 13 fisheye camera."""

    # Determine number of workers
    if workers is None:
        workers = multiprocessing.cpu_count()

    print("=" * 60)
    print("GoPro Hero 13 Fisheye Calibration (Parallel)")
    print("=" * 60)
    print(f"Video: {video}")
    print(f"Pattern: {pattern}")
    print(f"Pattern size: {cols}x{rows}")
    print(f"Square size: {square_size}m ({square_size*1000:.1f}mm)")
    print(f"Workers: {workers} parallel processes")

    # Create debug directory
    debug_path = None
    if not no_debug:
        debug_path = Path(debug_dir)
        debug_path.mkdir(exist_ok=True)
        print(f"Debug images: {debug_path}")
    print()

    if pattern == 'charuco':
        print("Using proper ChArUco detection with ArUco markers")
        print()

    # Load video to get info and extract frames
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Processing every {frame_skip} frames")
    print()

    # Extract frames to process
    print("Extracting frames for parallel processing...")
    frames_to_process = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames
        if frame_count % frame_skip != 0:
            continue

        frames_to_process.append((frame.copy(), frame_count))

    cap.release()

    print(f"✓ Extracted {len(frames_to_process)} frames")
    print()

    # Get image dimensions
    h, w = frames_to_process[0][0].shape[:2]

    # Prepare arguments for parallel processing
    print("Detecting calibration pattern in parallel...")
    print("(Processing on multiple CPU cores...)")
    print()

    objpoints = []
    imgpoints = []
    successful_frames = 0

    if pattern == 'checkerboard':
        # Prepare checkerboard object points
        objp = np.zeros((rows * cols, 3), np.float64)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= square_size

        # Prepare arguments for workers
        args_list = [(frame, frame_num, cols, rows, debug_path)
                     for frame, frame_num in frames_to_process]

        # Process in parallel
        with multiprocessing.Pool(workers) as pool:
            results = pool.map(process_frame_checkerboard, args_list)

        # Collect results
        for success, corners, debug_img, frame_num in results:
            if success:
                objpoints.append(objp.copy())
                imgpoints.append(corners)
                successful_frames += 1

                print(f"✓ Frame {frame_num}/{total_frames}: Pattern found ({successful_frames} total)")

                # Save debug image
                if debug_img is not None and debug_path is not None:
                    cv2.putText(debug_img, f"Frame {frame_num} - Pattern {successful_frames}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    debug_img_small = cv2.resize(debug_img, (1280, 960))
                    cv2.imwrite(str(debug_path / f"detected_{successful_frames:03d}.jpg"), debug_img_small)

    else:  # charuco
        # Prepare arguments for workers
        args_list = [(frame, frame_num, cols, rows, square_size, marker_size, debug_path)
                     for frame, frame_num in frames_to_process]

        # Process in parallel
        with multiprocessing.Pool(workers) as pool:
            results = pool.map(process_frame_charuco, args_list)

        # Collect results
        for success, obj_pts, img_pts, num_corners, debug_img, frame_num in results:
            if success:
                objpoints.append(obj_pts)
                imgpoints.append(img_pts)
                successful_frames += 1

                print(f"✓ Frame {frame_num}/{total_frames}: ChArUco found with {num_corners} corners ({successful_frames} total)")

                # Save debug image
                if debug_img is not None and debug_path is not None:
                    cv2.putText(debug_img, f"Frame {frame_num} - Pattern {successful_frames} ({num_corners} corners)",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    debug_img_small = cv2.resize(debug_img, (1280, 960))
                    cv2.imwrite(str(debug_path / f"detected_{successful_frames:03d}.jpg"), debug_img_small)

    print()
    print("=" * 60)
    print(f"Pattern Detection Complete")
    print(f"  Frames processed: {frame_count // frame_skip}")
    print(f"  Patterns found: {successful_frames}")
    print("=" * 60)
    print()

    if successful_frames < min_frames:
        print(f"✗ ERROR: Only found {successful_frames} frames with pattern.")
        print(f"  Need at least {min_frames} frames for reliable calibration.")
        print()
        print("Tips for better results:")
        print("  - Ensure good lighting (no shadows on pattern)")
        print("  - Move slower to avoid motion blur")
        print("  - Get more varied angles and distances")
        print("  - Keep entire pattern visible in frame")
        sys.exit(1)

    print(f"Image size: {w}x{h}")
    print()

    # Calibrate using fisheye model
    print("=" * 60)
    print("Calibrating FISHEYE model...")
    print("=" * 60)

    try:
        # Ensure proper data types for fisheye calibration
        K = np.zeros((3, 3), dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]

        # Convert lists to proper format for fisheye calibration
        if pattern == 'checkerboard':
            objpoints_array = [op.reshape(1, -1, 3).astype(np.float64) for op in objpoints]
            imgpoints_array = [ip.astype(np.float64) for ip in imgpoints]
        else:  # charuco - already in correct format from workers
            objpoints_array = objpoints
            imgpoints_array = imgpoints

        calibration_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_CHECK_COND +
            cv2.fisheye.CALIB_FIX_SKEW
        )

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints_array, imgpoints_array, (w, h), K, D, rvecs, tvecs,
            calibration_flags, criteria
        )

        print(f"✓ Fisheye calibration complete!")
        print(f"  RMS reprojection error: {rms:.4f} pixels")
        print()
        print("Camera Matrix (K):")
        print(K)
        print()
        print("Distortion Coefficients (D):")
        print(D.ravel())
        print()

        # Check quality
        if rms < 0.5:
            quality = "Excellent"
        elif rms < 1.0:
            quality = "Good"
        elif rms < 2.0:
            quality = "Acceptable"
        else:
            quality = "Poor - consider re-recording with better coverage"

        print(f"Calibration quality: {quality}")
        print()

        # Create output in UMI format
        output_data = {
            "final_reproj_error": float(rms),
            "fps": float(fps),
            "image_height": int(h),
            "image_width": int(w),
            "intrinsic_type": "FISHEYE",
            "intrinsics": {
                "focal_length": float((K[0, 0] + K[1, 1]) / 2),
                "aspect_ratio": float(K[1, 1] / K[0, 0]),
                "principal_pt_x": float(K[0, 2]),
                "principal_pt_y": float(K[1, 2]),
                "radial_distortion_1": float(D[0, 0]),
                "radial_distortion_2": float(D[1, 0]),
                "radial_distortion_3": float(D[2, 0]),
                "radial_distortion_4": float(D[3, 0]),
                "skew": 0.0
            },
            "nr_calib_images": successful_frames,
            "stabelized": False
        }

        # Save to file
        with open(output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print("=" * 60)
        print(f"✓ Calibration saved to: {output}")
        print("=" * 60)
        print()

        if not no_debug:
            print(f"Debug images saved to: {debug_dir}/")
            print(f"  Review detected patterns: {successful_frames} images")
            print()

        print("Next steps:")
        print(f"1. Copy to project: cp {output} example/calibration/")
        print("2. Update pipeline scripts to use this intrinsics file")
        print("3. Update SLAM mask dimensions to 4000x3000")
        print("4. Test SLAM on a short video")
        print()

        return 0

    except Exception as e:
        print(f"✗ Fisheye calibration failed: {e}")
        print()
        print("This might happen if:")
        print("  - Too few frames with good pattern detection")
        print("  - Pattern coverage is insufficient")
        print("  - Pattern size settings are incorrect")
        print()
        print("Try:")
        print("  - Recording a new calibration video with better coverage")
        print("  - Checking pattern dimensions (rows, cols, square_size)")
        print("  - Using --frame_skip 5 for more frames")
        sys.exit(1)


if __name__ == "__main__":
    main()
