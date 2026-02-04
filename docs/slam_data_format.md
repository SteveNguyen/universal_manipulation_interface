# SLAM Data Format Specification

This document details the exact data formats required by the ORB_SLAM3 fork for visual-inertial SLAM processing in the UMI pipeline.

**Note**: While the default pipeline uses GoPro cameras, the SLAM system can work with **any camera/IMU combination** that provides data in the required format.

## Overview

The SLAM system requires two synchronized data streams:
1. **Video** - MP4 video file (or image sequence)
2. **IMU Data** - JSON file containing accelerometer and gyroscope measurements

### Minimum Requirements for Custom Camera/IMU

To use a custom camera/IMU system, you need:

1. **Synchronized video and IMU data** with shared timestamps
2. **Camera intrinsic calibration** (focal length, principal point, distortion)
3. **IMU-to-camera extrinsic calibration** (rigid transform between sensors)
4. **IMU noise characterization** (noise densities, random walk)

---

## For Custom Camera/IMU Systems

### What You Need to Provide

| Component | Format | Description |
|-----------|--------|-------------|
| Video | MP4 or image sequence | Timestamped frames |
| IMU JSON | See format below | Accel + Gyro samples |
| Intrinsics | JSON or YAML | Camera calibration |
| Extrinsics | 4×4 matrix | IMU-to-camera transform |
| IMU noise | 4 parameters | From datasheet or Allan variance |

### Calibration Requirements

1. **Camera Intrinsics**: Use OpenCV, Kalibr, or OpenICC to calibrate
2. **IMU-Camera Extrinsics**: Use Kalibr or OpenICC with synchronized data
3. **IMU Noise**: From sensor datasheet or Allan variance analysis

---

## Critical Requirements Summary

### IMU Requirements

| Parameter | Requirement | Notes |
|-----------|-------------|-------|
| Sample Rate | ≥100 Hz (200 Hz recommended) | Higher is better for fast motion |
| Accelerometer | 3-axis, ±16g range typical | Must include gravity |
| Gyroscope | 3-axis, ±2000 dps typical | In rad/s for SLAM |
| Timestamp | Microsecond precision | Synchronized with video |
| Units | Accel: m/s², Gyro: rad/s | **Critical** - SLAM expects SI units |

### Video Requirements

| Parameter | Requirement | Notes |
|-----------|-------------|-------|
| Frame Rate | 30-60 Hz typical | Must be constant |
| Resolution | 640×480 to 1920×1080 | Higher needs more compute |
| Lens | Any (pinhole or fisheye) | Must be calibrated |
| Sync | Hardware sync preferred | Or software timestamp alignment |

### Coordinate System

The SLAM system uses the **IMU body frame** as the reference. You must provide:

1. **T_imu_cam**: 4×4 rigid transform from IMU frame to camera frame
2. This defines how to convert camera poses to IMU poses

---

## 1. Video Format

### File Requirements

| Property | Requirement | Notes |
|----------|-------------|-------|
| Container | MP4 | Required for GoPro GPMF metadata |
| Video Codec | H.264 or HEVC | Both supported via FFmpeg |
| Pixel Format | YUV420 | `yuvj420p` or `yuv420p` |
| Frame Rate | 50 or 60 FPS | Must match calibration FPS |
| Resolution | Configurable | See resolution options below |

### Supported Resolutions

| Resolution | Use Case | Notes |
|------------|----------|-------|
| 960×720 | Default SLAM | Best results, pre-downscaled |
| 2704×2028 | 2.7K pipeline | Alternative for comparison |
| 4000×3000 | Native 4K | Requires downscaling |

### Video Reading

The SLAM system reads video using OpenCV with FFmpeg backend:

```cpp
cv::VideoCapture cap(input_video, cv::CAP_FFMPEG);
```

**Important**: The video must be seekable and have correct frame count metadata.

### Example FFprobe Output

```json
{
  "streams": [
    {
      "codec_name": "hevc",
      "width": 960,
      "height": 720,
      "pix_fmt": "yuvj420p",
      "r_frame_rate": "60000/1001",
      "duration": "10.5"
    }
  ]
}
```

---

## 2. IMU Data Format

### File: `imu_data.json`

The IMU data is extracted from GoPro GPMF (GoPro Metadata Format) embedded in the MP4 file.

### JSON Structure

```json
{
  "frames/second": 59.94,
  "1": {
    "device name": "Hero13 Black",
    "streams": {
      "ACCL": {
        "name": "Accelerometer",
        "units": "m/s2",
        "samples": [
          {
            "value": [7.565, -1.292, 5.479],
            "cts": 20.18,
            "date": "2026-01-26T09:32:46.000Z",
            "temperature [°C]": 28.94
          }
        ]
      },
      "GYRO": {
        "name": "Gyroscope",
        "units": "rad/s",
        "samples": [
          {
            "value": [-0.038, 0.051, 0.003],
            "cts": 20.18,
            "date": "2026-01-26T09:32:46.000Z",
            "temperature [°C]": 28.94
          }
        ]
      },
      "CORI": {
        "name": "CameraOrientation",
        "units": "N/A",
        "samples": [...]
      },
      "IORI": {
        "name": "ImageOrientation",
        "units": "N/A",
        "samples": [...]
      },
      "GRAV": {
        "name": "Gravity Vector",
        "units": "N/A",
        "samples": [...]
      }
    }
  }
}
```

### Stream Types

| Stream | Name | Units | Usage | Rate |
|--------|------|-------|-------|------|
| `ACCL` | Accelerometer | m/s² | **Required** for SLAM | ~200 Hz |
| `GYRO` | Gyroscope | rad/s | **Required** for SLAM | ~200 Hz |
| `CORI` | Camera Orientation | quaternion | Optional (stabilization) | ~60 Hz |
| `IORI` | Image Orientation | quaternion | Optional | ~60 Hz |
| `GRAV` | Gravity Vector | normalized | Optional | ~60 Hz |

### Sample Format

Each IMU sample contains:

```json
{
  "value": [x, y, z],
  "cts": 1234.56,
  "date": "2026-01-26T09:32:46.000Z",
  "temperature [°C]": 28.94
}
```

| Field | Type | Description |
|-------|------|-------------|
| `value` | `[float, float, float]` | X, Y, Z measurement |
| `cts` | `float` | Timestamp in **milliseconds** from video start |
| `date` | `string` | ISO 8601 absolute timestamp (optional) |
| `temperature [°C]` | `float` | Sensor temperature (optional) |

### IMU Coordinate System

GoPro IMU follows a right-handed coordinate system:
- **X**: Right (when viewing camera from behind)
- **Y**: Up
- **Z**: Forward (optical axis direction)

**Accelerometer**: Reports specific force (includes gravity when stationary)
- At rest facing up: `[0, 0, +9.8]` (approximately)
- At rest facing down: `[0, 0, -9.8]`

**Gyroscope**: Reports angular velocity in rad/s
- Positive rotation follows right-hand rule

---

## 3. Timestamp Synchronization

### Video Timestamps (Computed, Not Read)

**Important**: The SLAM system does **not** read actual timestamps from video frames. Instead, timestamps are **computed** assuming constant frame rate:

```cpp
double fps = cap.get(cv::CAP_PROP_FPS);  // Get FPS from video metadata
double tframe = (double)frame_idx / fps;  // Compute timestamp

// Example at 60 FPS:
// Frame 0 → t = 0.0000s
// Frame 1 → t = 0.0167s
// Frame 2 → t = 0.0333s
// Frame 60 → t = 1.0000s
```

**Implications**:
- Video **must** have constant frame rate (CFR), not variable frame rate (VFR)
- Dropped frames in recording will cause timestamp drift
- The FPS metadata in the video file must be accurate

### IMU Timestamps (Normalized)

IMU timestamps are read from the JSON `cts` field (in milliseconds), then **normalized** so the first sample starts at t=0:

```cpp
// In LoadTelemetry():
double imu_start_t = sorted_acc.begin()->first;  // First IMU timestamp
for (auto acc : sorted_acc) {
    vTimeStamps.push_back(acc.first - imu_start_t);  // Normalize to start at 0
}
```

**Example**:
```
Original cts values:  20.18ms, 25.17ms, 30.15ms, 35.14ms...
After normalization:   0.00ms,  4.99ms,  9.97ms, 14.96ms...
```

### Synchronization Assumption

**Critical**: The system assumes video and IMU start at the same moment:

```
Time:        0.000s    0.017s    0.033s    0.050s    0.067s
             |         |         |         |         |
Video:       [Frame 0] [Frame 1] [Frame 2] [Frame 3] [Frame 4]
             |         |         |         |         |
IMU:         *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-->
             ↑
             Normalized to t=0
```

Both time bases are assumed to be aligned:
- Video frame 0 occurs at t=0
- First IMU sample (after normalization) is at t=0
- They represent the **same physical moment**

### Synchronization Algorithm

For each video frame at time `t_frame`:

1. Collect all IMU samples where `t_imu <= t_frame`
2. Package as `ORB_SLAM3::IMU::Point(acc, gyro, timestamp)`
3. Pass to SLAM tracking along with video frame

```cpp
// Main processing loop (simplified)
int last_imu_idx = 0;

for (frame_idx = 0; frame_idx < n_frames; frame_idx++) {
    double t_frame = (double)frame_idx / fps;

    // Collect all IMU measurements up to this frame's timestamp
    vector<IMU::Point> imu_measurements;
    while (last_imu_idx < n_imu && imuTimestamps[last_imu_idx] <= t_frame) {
        imu_measurements.push_back(IMU::Point(
            vAcc[last_imu_idx],      // cv::Point3f (m/s²)
            vGyr[last_imu_idx],      // cv::Point3f (rad/s)
            imuTimestamps[last_imu_idx]  // double (seconds)
        ));
        last_imu_idx++;
    }

    // Process frame with accumulated IMU data
    slam.TrackMonocular(frame, t_frame, imu_measurements);
}
```

### Expected IMU Rate

| Video FPS | IMU Rate | Samples/Frame | Notes |
|-----------|----------|---------------|-------|
| 60 Hz | 200 Hz | ~3.3 | GoPro typical |
| 50 Hz | 200 Hz | ~4.0 | PAL regions |
| 30 Hz | 200 Hz | ~6.7 | Lower-end cameras |
| 30 Hz | 100 Hz | ~3.3 | Minimum viable |

### What Happens If Timing Is Wrong?

| Problem | Symptom | Solution |
|---------|---------|----------|
| VFR video | Drift over time, tracking fails | Convert to CFR with ffmpeg |
| Wrong FPS metadata | Scale errors, IMU integration wrong | Re-encode with correct FPS |
| IMU starts late | First frames have no IMU, init fails | Trim video start or pad IMU |
| IMU starts early | Some IMU data unused (OK) | No action needed |
| Constant offset | Scale drift, poor tracking | Shift IMU timestamps |
| Clock drift | Gradual desync, tracking degrades | Resync clocks, use hardware sync |

### Ensuring Proper Alignment

For GoPro cameras, alignment is automatic because:
- Video and IMU are recorded in the same device
- GPMF metadata embeds IMU with video-relative timestamps
- Extraction preserves the relationship

For custom systems, you must ensure:

1. **Hardware sync** (best): Trigger camera and IMU from same signal
2. **Software sync**: Record a sharp motion event, align in post-processing
3. **Shared clock**: Both devices use same time source (NTP, PTP)

### Verifying Alignment

Test with a sharp motion (e.g., tap the camera):

```python
import json
import numpy as np

# Load IMU
with open('imu_data.json') as f:
    imu = json.load(f)

accl = imu['1']['streams']['ACCL']['samples']
timestamps = [s['cts']/1000 for s in accl]  # Convert to seconds
accel_mag = [np.linalg.norm(s['value']) for s in accl]

# Find spike in acceleration (the tap)
spike_idx = np.argmax(accel_mag)
spike_time = timestamps[spike_idx]

print(f"IMU spike at t={spike_time:.3f}s")
print(f"This should match the video frame where motion blur appears")
# video_frame = int(spike_time * fps)
```

---

## 4. SLAM Settings YAML

### Required Parameters

```yaml
%YAML:1.0

# Camera Model
Camera.type: "KannalaBrandt8"  # Fisheye model

# Intrinsics (resolution-dependent)
Camera1.fx: 371.35
Camera1.fy: 370.71
Camera1.cx: 479.0
Camera1.cy: 359.6

# Fisheye Distortion (k1-k4)
Camera1.k1: 0.1295
Camera1.k2: -0.1708
Camera1.k3: 0.1322
Camera1.k4: -0.0386

# Resolution
Camera.width: 960
Camera.height: 720
Camera.fps: 60
Camera.RGB: 1  # 0=BGR, 1=RGB

# IMU-to-Camera Transform (4x4 matrix)
IMU.T_b_c1: !!opencv-matrix
    rows: 4
    cols: 4
    dt: f
    data: [r11, r12, r13, tx,
           r21, r22, r23, ty,
           r31, r32, r33, tz,
           0.0, 0.0, 0.0, 1.0]

# IMU Noise Parameters
IMU.NoiseGyro: 0.0015      # rad/s^0.5
IMU.NoiseAcc: 0.017        # m/s^1.5
IMU.GyroWalk: 5.0e-5       # rad/s^1.5
IMU.AccWalk: 0.0055        # m/s^2.5
IMU.Frequency: 200.0       # Hz

# ORB Feature Extraction
ORBextractor.nFeatures: 1250
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

# Far Point Filtering
System.thFarPoints: 20.0   # meters
```

### Parameter Descriptions

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `Camera.type` | Camera model | `KannalaBrandt8` (fisheye) |
| `Camera1.fx/fy` | Focal length in pixels | Resolution-dependent |
| `Camera1.cx/cy` | Principal point | ~center of image |
| `Camera1.k1-k4` | Fisheye distortion | From calibration |
| `IMU.T_b_c1` | IMU-to-camera transform | From calibration |
| `IMU.NoiseGyro` | Gyro white noise density | ~0.001-0.002 |
| `IMU.NoiseAcc` | Accel white noise density | ~0.01-0.02 |
| `IMU.Frequency` | IMU sample rate | 200 Hz for GoPro |

---

## 5. Extracting IMU Data

### From GoPro Videos

Use the OpenICC docker image:

```bash
docker run --rm \
    -v /path/to/video_dir:/data \
    chicheng/openicc:latest \
    node /OpenImuCameraCalibrator/javascript/extract_metadata_single.js \
    /data/raw_video.mp4 \
    /data/imu_data.json
```

### Pipeline Script

```bash
python scripts_slam_pipeline/01_extract_gopro_imu.py /path/to/session
```

This processes all videos in `session/demos/*/raw_video.mp4` and creates `imu_data.json` for each.

---

## 6. Data Validation

### Video Checks

```bash
# Check resolution and FPS
ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,r_frame_rate \
    -of csv=p=0 raw_video.mp4
# Expected: 960,720,60000/1001
```

### IMU Checks

```python
import json

with open('imu_data.json') as f:
    data = json.load(f)

# Check FPS metadata
print(f"Video FPS: {data.get('frames/second', 'N/A')}")

# Check stream availability
frame = data['1']
streams = frame.get('streams', {})
print(f"ACCL samples: {len(streams.get('ACCL', {}).get('samples', []))}")
print(f"GYRO samples: {len(streams.get('GYRO', {}).get('samples', []))}")

# Check timestamp range
accl_samples = streams['ACCL']['samples']
first_ts = accl_samples[0]['cts']
last_ts = accl_samples[-1]['cts']
print(f"Time range: {first_ts:.2f}ms to {last_ts:.2f}ms")
print(f"Duration: {(last_ts - first_ts) / 1000:.2f}s")
```

### Expected Values

| Check | Expected |
|-------|----------|
| ACCL samples | ~200 × video_duration_sec |
| GYRO samples | ~200 × video_duration_sec |
| Sample interval | ~5ms (200 Hz) |
| Timestamp alignment | IMU starts at ~0, video frames at 0 |

---

## 7. Common Issues

### Missing IMU Data

**Symptoms**: `imu_data.json` is empty or missing streams

**Causes**:
- GoPro stabilization was ON (strips raw IMU)
- Video was transcoded/re-encoded (strips GPMF metadata)
- Incompatible GoPro model

**Solution**: Re-record with stabilization OFF, use original MP4 files

### Timestamp Mismatch

**Symptoms**: SLAM tracking fails, scale issues

**Causes**:
- Video FPS doesn't match calibration FPS
- IMU timestamps corrupted

**Solution**: Verify FPS matches calibration, re-extract IMU data

### Resolution Mismatch

**Symptoms**: Distorted features, tracking drift

**Causes**:
- Using wrong intrinsics for video resolution
- Video was resized without updating intrinsics

**Solution**: Use matching intrinsics file, or regenerate SLAM settings

---

## 8. File Organization

Expected directory structure for SLAM processing:

```
session/
└── demos/
    ├── mapping/
    │   ├── raw_video.mp4         # Video file
    │   ├── imu_data.json         # IMU data
    │   ├── slam_settings_auto.yaml  # Auto-generated
    │   ├── map_atlas.osa         # SLAM map output
    │   └── mapping_camera_trajectory.csv  # Trajectory output
    └── demo_*/
        ├── raw_video.mp4
        ├── imu_data.json
        └── camera_trajectory.csv
```

---

## 9. Custom Camera/IMU Integration Guide

### Step-by-Step Process

#### 1. Prepare Your Hardware

- Camera with known/calibratable lens model
- IMU with accelerometer + gyroscope (e.g., BMI160, ICM-20948, MPU-9250)
- Rigid mounting between camera and IMU
- Hardware or software timestamp synchronization

#### 2. Record Synchronized Data

Your recording system must:
1. Capture video frames with timestamps
2. Record IMU at ≥100 Hz with timestamps
3. Use the same time base for both (or convert later)

#### 3. Calibrate Camera Intrinsics

Use any standard calibration tool:

```bash
# OpenCV calibration
python calibrate_camera.py --board chessboard --size 9x6 --square_mm 25

# Kalibr
kalibr_calibrate_cameras --bag calib.bag --topics /cam0/image_raw --models pinhole-radtan

# OpenICC
python calibrate_camera.py --input video.mp4 --board charuco
```

Output format needed:
```json
{
  "image_width": 960,
  "image_height": 720,
  "intrinsic_type": "FISHEYE",
  "intrinsics": {
    "focal_length": 371.35,
    "aspect_ratio": 1.0,
    "principal_pt_x": 479.0,
    "principal_pt_y": 359.6,
    "radial_distortion_1": 0.1295,
    "radial_distortion_2": -0.1708,
    "radial_distortion_3": 0.1322,
    "radial_distortion_4": -0.0386
  }
}
```

#### 4. Calibrate IMU-Camera Extrinsics

Use Kalibr or OpenICC:

```bash
# Kalibr
kalibr_calibrate_imu_camera --bag calib.bag \
    --cam cam_chain.yaml \
    --imu imu.yaml

# OpenICC
python calibrate_imu_camera.py --video cam_imu_video.mp4
```

Output needed: 4×4 transformation matrix T_imu_cam

#### 5. Get IMU Noise Parameters

From datasheet or Allan variance analysis:

| Parameter | Symbol | Typical Range | Units |
|-----------|--------|---------------|-------|
| Gyro noise density | `IMU.NoiseGyro` | 0.001-0.01 | rad/s/√Hz |
| Accel noise density | `IMU.NoiseAcc` | 0.01-0.1 | m/s²/√Hz |
| Gyro random walk | `IMU.GyroWalk` | 1e-5 to 1e-4 | rad/s²/√Hz |
| Accel random walk | `IMU.AccWalk` | 0.001-0.01 | m/s³/√Hz |

**From datasheet**: Look for "noise density" or "spectral noise density"

**From Allan variance**: Use imu_utils or kalibr_allan

#### 6. Create IMU JSON File

Convert your IMU data to the required format:

```python
import json

def create_imu_json(accel_data, gyro_data, output_path):
    """
    Create imu_data.json from custom IMU data.

    Args:
        accel_data: List of (timestamp_ms, [ax, ay, az]) tuples
                   Units: timestamp in ms, acceleration in m/s²
        gyro_data: List of (timestamp_ms, [gx, gy, gz]) tuples
                  Units: timestamp in ms, angular velocity in rad/s
    """
    accl_samples = [
        {"value": list(acc), "cts": ts}
        for ts, acc in accel_data
    ]

    gyro_samples = [
        {"value": list(gyro), "cts": ts}
        for ts, gyro in gyro_data
    ]

    imu_json = {
        "frames/second": 60.0,  # Your video FPS
        "1": {
            "streams": {
                "ACCL": {
                    "name": "Accelerometer",
                    "units": "m/s2",
                    "samples": accl_samples
                },
                "GYRO": {
                    "name": "Gyroscope",
                    "units": "rad/s",
                    "samples": gyro_samples
                }
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(imu_json, f, indent=2)
```

**Critical**:
- Timestamps in **milliseconds** from video start
- Accelerometer in **m/s²** (includes gravity)
- Gyroscope in **rad/s** (not deg/s!)

#### 7. Create SLAM Settings YAML

```yaml
%YAML:1.0

# Camera Model (choose one)
Camera.type: "PinHole"       # For standard lenses
# Camera.type: "KannalaBrandt8"  # For fisheye lenses

# Your calibrated intrinsics
Camera1.fx: 500.0
Camera1.fy: 500.0
Camera1.cx: 320.0
Camera1.cy: 240.0

# Distortion (for PinHole: k1,k2,p1,p2,k3; for Fisheye: k1,k2,k3,k4)
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.k3: 0.0
Camera1.k4: 0.0

# Resolution
Camera.width: 640
Camera.height: 480
Camera.fps: 30
Camera.RGB: 1

# IMU-to-Camera transform (your calibration result)
IMU.T_b_c1: !!opencv-matrix
    rows: 4
    cols: 4
    dt: f
    data: [1, 0, 0, 0,     # Replace with your
           0, 1, 0, 0,     # calibrated values
           0, 0, 1, 0,
           0, 0, 0, 1]

# IMU noise (from datasheet or Allan variance)
IMU.NoiseGyro: 0.004       # Your value
IMU.NoiseAcc: 0.05         # Your value
IMU.GyroWalk: 2.0e-5       # Your value
IMU.AccWalk: 0.002         # Your value
IMU.Frequency: 200.0       # Your IMU rate

# ORB Parameters (can usually keep defaults)
ORBextractor.nFeatures: 1250
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
```

#### 8. Run SLAM

```bash
docker run --rm \
    -v /path/to/data:/data \
    chicheng/orb_slam3:latest \
    /ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam \
    -i /data/video.mp4 \
    -j /data/imu_data.json \
    -s /data/custom_settings.yaml \
    -o /data/trajectory.csv
```

### Common Conversion Issues

#### IMU Units

| Source | Convert to |
|--------|------------|
| mg (milli-g) | m/s² (multiply by 9.81/1000) |
| g | m/s² (multiply by 9.81) |
| dps (deg/s) | rad/s (multiply by π/180) |
| LSB | Check datasheet for conversion |

#### Timestamp Alignment

If camera and IMU have different time bases:

```python
# Find time offset using correlation or manual alignment
time_offset = find_offset(camera_timestamps, imu_timestamps)

# Apply offset to IMU timestamps
imu_timestamps_aligned = imu_timestamps + time_offset
```

#### Coordinate Frame Conversion

Common IMU axes conventions:
- **NED** (North-East-Down): aerospace standard
- **ENU** (East-North-Up): robotics common
- **FRD** (Forward-Right-Down): some cameras

Convert to match your camera frame:
```python
# Example: NED to camera (X-right, Y-down, Z-forward)
acc_cam = [acc_ned[1], acc_ned[2], acc_ned[0]]
gyro_cam = [gyro_ned[1], gyro_ned[2], gyro_ned[0]]
```

### Validating Your Setup

1. **Static test**: Place camera still, check accel magnitude ≈ 9.81 m/s²
2. **Rotation test**: Rotate 90°, check gyro integration matches
3. **Sync test**: Sharp motion should align in video and IMU
4. **SLAM test**: Should initialize and track for simple motion

---

## References

- [GoPro GPMF Parser](https://github.com/gopro/gpmf-parser) - GoPro metadata format
- [OpenImuCameraCalibrator](https://github.com/urbste/OpenImuCameraCalibrator) - IMU extraction and calibration
- [Kalibr](https://github.com/ethz-asl/kalibr) - Camera-IMU calibration
- [ORB_SLAM3 UMI Fork](https://github.com/cheng-chi/ORB_SLAM3) - Modified SLAM system
- [Kannala-Brandt Model](https://ieeexplore.ieee.org/document/1642666) - Fisheye camera model
- [imu_utils](https://github.com/gaowenliang/imu_utils) - Allan variance analysis
