# ORB_SLAM3 Fork Analysis

This document explains the modifications made to ORB_SLAM3 for GoPro support in the UMI pipeline.

## Repository Links

- **UMI Fork**: https://github.com/cheng-chi/ORB_SLAM3
- **Original ORB_SLAM3**: https://github.com/UZ-SLAMLab/ORB_SLAM3
- **Docker Image**: https://hub.docker.com/r/chicheng/orb_slam3

## Overview

The fork by Cheng Chi transforms the original ORB_SLAM3 from an academic SLAM library into a production-ready tool specifically for the UMI (Universal Manipulation Interface) pipeline. It adds GoPro camera support, telemetry parsing, gripper masking, and batch processing capabilities.

---

## 1. New GoPro-Specific Executables

**Added files in `Examples/Monocular-Inertial/`:**

| File | Purpose |
|------|---------|
| `gopro_slam.cc` | Main SLAM executable for mapping and localization |
| `mono_inertial_gopro_vi.cc` | Visual-inertial processing for GoPro |
| `mono_inertial_gopro_localize.cc` | Localization-only mode against existing map |

### Key Features of `gopro_slam.cc`

- **Video Input**: Reads video files directly via OpenCV `VideoCapture`
- **IMU Integration**: Loads accelerometer/gyroscope data from GoPro GPMF telemetry (JSON format)
- **Timestamp Synchronization**: Synchronizes IMU measurements with video frames
- **Image Masking**: Applies masks to exclude gripper/mirrors from feature detection
- **ArUco Initialization**: Uses ArUco markers for precise starting pose with known scale
- **Map Persistence**: Save and load SLAM maps for localization across sessions
- **Trajectory Export**: CSV output with frame indices, timestamps, poses, and keyframe flags
- **Early Termination**: Stops when tracking is lost for too many consecutive frames (`--max_lost_frames`)

### Command Line Arguments

```
-i/--input_video        Video file path (required)
-j/--input_imu_json     GoPro telemetry JSON file (required)
-s/--setting            Camera calibration YAML file
-o/--output_trajectory_csv  Output trajectory path
--init_tag_id           ArUco tag ID for initialization
--init_tag_size         Physical size of initialization tag (meters)
-g/--enable_gui         Enable visualization
--max_lost_frames       Terminate if tracking lost beyond threshold
--load_map              Load existing map for localization
--save_map              Save map after processing
```

---

## 2. GoPro Camera Configurations

**New YAML configuration files:**

| File | Resolution | Camera |
|------|------------|--------|
| `gopro10_maxlens_fisheye_setting_v1_720.yaml` | 960×720 | GoPro 10 MaxLens |
| `gopro10_maxlens_fisheye_setting_v1_480.yaml` | 640×480 | GoPro 10 MaxLens |
| `gopro10_maxlens_fisheye_setting_v1.yaml` | 2704×2028 | GoPro 10 MaxLens |
| `gopro9_maxlens_fisheye_setting.yaml` | - | GoPro 9 MaxLens |
| `gopro9_wide_setting.yaml` | - | GoPro 9 Wide |

### Configuration Parameters

```yaml
# Camera Model
Camera.type: "KannalaBrandt8"  # Fisheye with 4 distortion coefficients

# Resolution and FPS
Camera.width: 960
Camera.height: 720
Camera.fps: 60

# Intrinsics (example for 720p)
Camera1.fx: 282.91
Camera1.fy: 282.91
Camera1.cx: 480.0
Camera1.cy: 360.0

# Fisheye Distortion (k1-k4)
Camera1.k1: -0.0175
Camera1.k2: -0.0281
Camera1.k3: 0.0128
Camera1.k4: -0.0045

# IMU Parameters (GoPro specific)
IMU.Frequency: 200          # Hz
IMU.NoiseGyro: 0.0015       # rad/s^0.5
IMU.NoiseAcc: 0.017         # m/s^1.5
IMU.GyroWalk: 5.0e-5        # rad/s^1.5
IMU.AccWalk: 0.0055         # m/s^2.5

# IMU-to-Camera Transform (T_b_c)
IMU.T_b_c1: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [...]  # 4x4 transformation matrix

# ORB Feature Extraction
ORBextractor.nFeatures: 1250
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

# Far Point Filtering
System.thFarPoints: 20.0    # meters
```

---

## 3. Core Algorithm Modifications

### ArUco Marker Initialization

The original ORB_SLAM3 uses automatic initialization which can have scale ambiguity in monocular mode. The fork adds ArUco-based initialization:

- ArUco dictionary configurable via command line
- Tag-based reconstruction in `TwoViewReconstruction` module
- Known physical tag size provides metric scale
- Critical for accurate gripper pose estimation

### Tracking Improvements

| Feature | Original | Fork |
|---------|----------|------|
| Lost frame handling | Continue until end | Early termination after N lost frames |
| Far point filtering | Not configurable | `System.thFarPoints` parameter |
| Image masking | Polygon parameters | Image mask input |
| IMU preintegration | Had segfaults | Fixed |

### Output Enhancements

**CSV Trajectory Format:**
```csv
frame_idx,timestamp,x,y,z,q_x,q_y,q_z,q_w,is_lost,is_keyframe
0,0.0,1.234,2.345,3.456,0.0,0.0,0.0,1.0,0,1
1,0.0167,1.235,2.346,3.457,0.001,0.002,0.003,0.999,0,0
...
```

This format enables:
- Frame-accurate trajectory analysis
- Tracking quality metrics (lost frame counting)
- Keyframe identification for downstream processing

---

## 4. Code Cleanup & Robustness

### Bug Fixes
- Fixed segmentation faults in IMU preintegration handling
- Corrected inverted conditional logic in tracking state management
- Resolved compilation errors with modern compilers
- Docker signal handling fixes (proper SIGTERM handling)

### Code Quality
- Removed timing preprocessor directives
- Removed unused example code (EuRoC, TUM-VI executables kept but GoPro added)
- Converted assertions to exceptions for proper program termination
- Better video frame reading (captures every frame reliably)

---

## 5. Docker Integration

The fork includes a `Dockerfile` for containerized deployment:

```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get install -y \
    build-essential \
    libopencv-dev \
    libboost-dev \
    libboost-serialization-dev \
    libssl-dev \
    # Pangolin dependencies...

# Copy and build ORB_SLAM3
COPY . /ORB_SLAM3
WORKDIR /ORB_SLAM3
RUN ./build.sh
```

**Docker image contents:**
- Ubuntu 22.04 base
- Pangolin visualization library
- OpenCV 4.x
- Boost serialization
- Pre-compiled ORB_SLAM3 with all GoPro examples

**Usage in UMI pipeline:**
```bash
docker run -v /data:/data chicheng/orb_slam3 \
    /ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam \
    -i /data/video.mp4 \
    -j /data/imu_data.json \
    -s /ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml \
    -o /data/trajectory.csv
```

---

## 6. Comparison: Original vs Fork

| Aspect | Original ORB_SLAM3 | UMI Fork |
|--------|-------------------|----------|
| **Input Sources** | Live cameras, image sequences | Video files + JSON telemetry |
| **Supported Cameras** | EuRoC, TUM-VI, RealSense | GoPro 9/10/11/13 (fisheye) |
| **IMU Data Format** | Binary/CSV datasets | GoPro GPMF JSON |
| **Initialization** | Automatic (scale ambiguous) | ArUco tag-based (known scale) |
| **Image Masking** | Not supported | Gripper/mirror exclusion |
| **Output Format** | TUM format only | TUM + CSV with frame indices |
| **Batch Processing** | Manual | Automated with early termination |
| **Deployment** | Build from source | Docker container |
| **Error Handling** | Assertions | Exceptions |

---

## 7. Why These Changes Matter for UMI

### 1. GoPro Telemetry Parsing
GoPro cameras embed IMU data in MP4 files using the GPMF (GoPro Metadata Format). The fork extracts this into JSON and synchronizes it with video frames, eliminating the need for external IMU hardware.

### 2. Gripper Masking
The UMI gripper appears in the camera's field of view. Without masking, ORB features would be detected on the gripper itself, causing the SLAM system to track the robot rather than the environment. The fork accepts mask images to exclude these regions.

### 3. ArUco Initialization
The ArUco marker provides:
- Known metric scale (critical for monocular SLAM)
- Consistent world coordinate frame across all demos
- Reference point for gripper-to-world calibration

### 4. Batch Processing Support
The `--max_lost_frames` parameter and CSV output enable:
- Automated pipeline processing of many demos
- Quality metrics for filtering bad recordings
- Early termination to avoid wasting compute on failed tracking

### 5. Docker Deployment
Containerization ensures:
- Reproducible SLAM results across different machines
- No dependency conflicts with host system
- Easy deployment in cloud/cluster environments

---

## 8. File Structure in Fork

```
ORB_SLAM3/
├── Examples/
│   └── Monocular-Inertial/
│       ├── gopro_slam.cc                          # Main GoPro SLAM executable
│       ├── mono_inertial_gopro_vi.cc              # VI processing
│       ├── mono_inertial_gopro_localize.cc        # Localization mode
│       ├── gopro10_maxlens_fisheye_setting_v1_720.yaml
│       ├── gopro10_maxlens_fisheye_setting_v1_480.yaml
│       ├── gopro10_maxlens_fisheye_setting_v1.yaml
│       ├── gopro9_maxlens_fisheye_setting.yaml
│       └── gopro9_wide_setting.yaml
├── src/
│   └── TwoViewReconstruction.cc                   # Modified for ArUco init
├── include/
│   └── ...
├── Dockerfile
└── build.sh
```

---

## 9. References

- [ORB-SLAM3 Paper](https://arxiv.org/abs/2007.11898) - Original algorithm
- [GoPro GPMF Spec](https://github.com/gopro/gpmf-parser) - Telemetry format
- [Kannala-Brandt Model](https://ieeexplore.ieee.org/document/1642666) - Fisheye camera model
- [UMI Project](https://umi-gripper.github.io/) - Universal Manipulation Interface
