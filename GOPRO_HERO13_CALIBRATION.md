# GoPro Hero 13 Ultra-Wide Calibration Guide

## Understanding Your Camera Setup

The GoPro Hero 13 with ultra-wide lens is different from the Hero 10 MaxLens fisheye:

- **Hero 10 MaxLens**: Extreme fisheye lens with circular distortion (Kannala-Brandt fisheye model)
- **Hero 13 Ultra-Wide**: Less extreme wide-angle, may use perspective distortion model

The key difference: If your image fills the rectangular frame without circular edges, it's likely using a **perspective camera model** rather than a fisheye model.

## Step 1: Verify Your Camera Settings

First, check your Hero 13 video settings:

1. **Resolution**: Check what resolution you're recording at (e.g., 5.3K, 4K, 2.7K)
2. **Lens Mode**: Ultra-Wide (you mentioned this)
3. **Frame Rate**: Ideally 60fps or higher for better SLAM performance
4. **Hypersmooth**: Turn OFF (adds digital stabilization which changes intrinsics)
5. **GoPro Labs Firmware**: Confirm it's installed and IMU recording is enabled

## Step 2: Test IMU Data Extraction

Before calibrating, verify that IMU data extraction works with your Hero 13:

```bash
# Record a short test video with your Hero 13
# Copy it to a test directory
mkdir -p test_hero13/demos/test_video
cp /path/to/your/GX010001.MP4 test_hero13/demos/test_video/raw_video.mp4

# Try extracting IMU data
source .venv/bin/activate
python scripts_slam_pipeline/01_extract_gopro_imu.py test_hero13
```

**Check the output**:
```bash
# Should create: test_hero13/demos/test_video/imu_data.json
cat test_hero13/demos/test_video/imu_data.json | head -100
```

If this works, your Hero 13 IMU data format is compatible! If not, we'll need to debug the extraction.

## Step 3: Camera Calibration Options

You have two approaches:

### Option A: Use OpenImuCameraCalibrator (Recommended)

This is the same tool used for the Hero 10, and it supports both fisheye and perspective models.

**Requirements**:
- Calibration pattern (checkerboard or ChArUco board)
- Well-lit environment
- Record video moving around the pattern

**Steps**:

1. **Download/use the Docker image**:
   ```bash
   docker pull chicheng/openicc:latest
   ```

2. **Print a calibration pattern**:
   - Download ChArUco board: https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html
   - Or use checkerboard pattern (9x6 is common)
   - Print on flat, rigid surface (foam board recommended)

3. **Record calibration video**:
   - Record 30-60 seconds of video with your Hero 13
   - Move around the pattern from different angles
   - Get close and far views
   - Keep pattern in frame entire time
   - Ensure good lighting, no motion blur

4. **Run calibration** (this requires setting up the OpenImuCameraCalibrator tool):
   - Follow: https://github.com/urbste/OpenImuCameraCalibrator/
   - The tool will auto-detect if fisheye or perspective model fits better
   - Output will be a JSON file similar to `gopro_intrinsics_2_7k.json`

### Option B: Quick Test with OpenCV Calibration

If you want to quickly test, you can use OpenCV's calibration:

```python
# Save this as calibrate_hero13.py
import cv2
import numpy as np
import json
import glob

# Checkerboard dimensions (inner corners)
CHECKERBOARD = (9, 6)  # Adjust to your pattern
square_size = 0.025  # 25mm squares (adjust to your pattern)

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# Load video
video_path = 'path/to/calibration_video.mp4'
cap = cv2.VideoCapture(video_path)

frame_count = 0
successful_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Use every 10th frame to avoid too similar views
    if frame_count % 10 != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)

        # Refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        successful_frames += 1

        print(f"Frame {frame_count}: Found pattern ({successful_frames} total)")

cap.release()

if successful_frames < 10:
    print(f"Error: Only found {successful_frames} frames with pattern. Need at least 10.")
    exit(1)

print(f"\nCalibrating with {successful_frames} frames...")

# Get image dimensions
h, w = gray.shape[:2]

# Try perspective model first (more common for ultra-wide)
print("\n=== Testing PERSPECTIVE model ===")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (w, h), None, None)

print(f"Perspective RMS reprojection error: {ret:.4f} pixels")
print(f"Camera matrix:\n{mtx}")
print(f"Distortion coefficients: {dist}")

# Also try fisheye model
print("\n=== Testing FISHEYE model ===")
try:
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs_fish = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
    tvecs_fish = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]

    ret_fish, K, D, rvecs_fish, tvecs_fish = cv2.fisheye.calibrate(
        objpoints, imgpoints, (w, h), K, D, rvecs_fish, tvecs_fish,
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

    print(f"Fisheye RMS reprojection error: {ret_fish:.4f} pixels")
    print(f"Camera matrix:\n{K}")
    print(f"Distortion coefficients: {D.ravel()}")
except Exception as e:
    print(f"Fisheye calibration failed: {e}")
    ret_fish = float('inf')

# Recommend which model to use
if ret < ret_fish and ret < 1.0:
    print("\n✓ RECOMMENDATION: Use PERSPECTIVE model")
    print(f"  Reason: Lower error ({ret:.4f} vs {ret_fish:.4f}) and good fit")
    model_type = "perspective"
elif ret_fish < ret and ret_fish < 1.0:
    print("\n✓ RECOMMENDATION: Use FISHEYE model")
    print(f"  Reason: Lower error ({ret_fish:.4f} vs {ret:.4f}) and good fit")
    model_type = "fisheye"
else:
    print("\n⚠ WARNING: Both models have high error")
    print("  Consider recording better calibration video")
    model_type = "perspective"  # Default

# Get frame rate from video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

# Save result in UMI format
if model_type == "perspective":
    output = {
        "final_reproj_error": float(ret),
        "fps": fps,
        "image_height": h,
        "image_width": w,
        "intrinsic_type": "PERSPECTIVE",
        "intrinsics": {
            "fx": float(mtx[0, 0]),
            "fy": float(mtx[1, 1]),
            "cx": float(mtx[0, 2]),
            "cy": float(mtx[1, 2]),
            "k1": float(dist[0, 0]),
            "k2": float(dist[0, 1]),
            "p1": float(dist[0, 2]),
            "p2": float(dist[0, 3]),
            "k3": float(dist[0, 4]) if dist.shape[1] > 4 else 0.0
        }
    }
else:  # fisheye
    output = {
        "final_reproj_error": float(ret_fish),
        "fps": fps,
        "image_height": h,
        "image_width": w,
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
        }
    }

output_file = "hero13_ultrawide_intrinsics.json"
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Calibration saved to: {output_file}")
print("\nNext steps:")
print("1. Copy this file to: example/calibration/")
print("2. Update SLAM pipeline to use this intrinsics file")
print("3. If PERSPECTIVE model, you'll need to modify ORB_SLAM3 settings")
```

## Step 4: Determine Camera Model Type

Based on your observation that it's "not extreme fisheye", you likely have one of these:

1. **Perspective/Rectilinear**: Straight lines stay straight, less distortion at edges
   - Common for "Ultra-Wide" mode on newer GoPros
   - Uses radial-tangential distortion (Brown-Conrady model)
   - OpenCV standard calibration

2. **Mild Fisheye**: Some barrel distortion but not extreme
   - Still uses fisheye model but with smaller distortion coefficients
   - Kannala-Brandt model

**Visual Test**:
- Record a video of a scene with straight lines (door frames, tiles, building edges)
- Look at the edges of the frame:
  - **Straight lines**: Perspective model
  - **Curved lines**: Fisheye model

## Step 5: Modify the Pipeline

### If Your Hero 13 Uses PERSPECTIVE Model:

This requires more changes because ORB_SLAM3 is configured for fisheye:

1. **Update intrinsics file**:
   ```bash
   cp hero13_ultrawide_intrinsics.json example/calibration/
   ```

2. **Create new ORB_SLAM3 settings file**:
   You'll need to create a perspective camera config. The fisheye config won't work.

   **Problem**: The provided ORB_SLAM3 Docker image only has fisheye configs. You may need to:
   - Create a custom ORB_SLAM3 config for perspective cameras
   - Or use a different SLAM system that supports perspective better
   - Or use the fisheye mode anyway (might still work with mild distortion)

3. **Update SLAM script**:
   ```bash
   # Edit scripts_slam_pipeline/02_create_map.py:80
   # Change from fisheye settings to perspective settings
   ```

### If Your Hero 13 Uses FISHEYE Model (with less distortion):

This is easier - just update the intrinsics:

1. **Copy your calibration**:
   ```bash
   cp hero13_ultrawide_intrinsics.json example/calibration/gopro_hero13_intrinsics.json
   ```

2. **Update processing scripts to use new intrinsics**:
   ```bash
   # In scripts_slam_pipeline/04_detect_aruco.py and other scripts
   # Change the default intrinsics path
   ```

3. **Update mask dimensions**:
   ```python
   # In scripts_slam_pipeline/02_create_map.py:62
   # Change (2028, 2704) to your Hero 13 resolution
   slam_mask = np.zeros((YOUR_HEIGHT, YOUR_WIDTH), dtype=np.uint8)
   ```

## Step 6: Test End-to-End

Once calibrated:

```bash
# 1. Record a test video with your Hero 13
# 2. Create test directory
mkdir -p test_hero13_session/raw_videos
cp YOUR_VIDEO.MP4 test_hero13_session/raw_videos/

# 3. Update intrinsics paths in the pipeline scripts
# Edit: scripts_slam_pipeline/04_detect_aruco.py
# Change: --intrinsics_json path

# 4. Run the pipeline
python run_slam_pipeline.py test_hero13_session
```

## Recommendations

Given your situation, I recommend:

1. **First**: Test IMU extraction to confirm Hero 13 compatibility
2. **Second**: Use the quick OpenCV calibration script to determine if it's perspective or fisheye
3. **Third**: If perspective, consider these options:
   - **Option A**: Use a different SLAM system (like COLMAP or Stella VSLAM) that handles perspective better
   - **Option B**: Modify ORB_SLAM3 config for perspective cameras (more work)
   - **Option C**: Stick with fisheye mode - it might work with mild distortion
4. **Fourth**: If fisheye (just milder), update intrinsics and proceed

## Expected Challenges

- **ORB_SLAM3 configuration**: The Docker image is set up for fisheye; perspective needs different settings
- **Resolution differences**: Hero 13 might record at different resolutions than Hero 10
- **IMU calibration**: IMU-to-camera extrinsics might differ
- **Distortion model mismatch**: If using wrong model, SLAM will fail or be inaccurate

## Next Steps

Let me know:
1. Can you extract IMU data from your Hero 13 video? (Step 2)
2. What resolution is your Hero 13 recording? (check video file properties)
3. After running the calibration script, which model (perspective or fisheye) has lower error?

Based on your answers, I can provide more specific guidance!
