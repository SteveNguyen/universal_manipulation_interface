# Camera Requirements and GoPro-Specific Components

This document explains what is specific to GoPro Hero cameras in the UMI project and what you need to adapt for different cameras.

## GoPro-Specific Requirements

### 1. **GoPro Labs Firmware** (CRITICAL)
- **Location**: README.md:130
- **Purpose**: Enables high-frequency IMU data recording embedded in MP4 metadata
- **Requirement**: GoPro must have [GoPro Labs firmware](https://gopro.com/en/us/info/gopro-labs) installed
- **Why it matters**: Standard GoPro firmware doesn't record IMU data at the required frequency

### 2. **Camera Resolution and Intrinsics**
- **Default Resolution**: 2704 x 2028 pixels (2.7K mode)
- **Intrinsics File**: `example/calibration/gopro_intrinsics_2_7k.json`
- **Lens Type**: Fisheye (radial distortion model with 4 coefficients)
- **FPS**: 59.94 fps (NTSC) - critical for timing synchronization

**Intrinsics Parameters**:
```json
{
  "fps": 59.94005994005994,
  "image_height": 2028,
  "image_width": 2704,
  "intrinsic_type": "FISHEYE",
  "intrinsics": {
    "focal_length": 796.8544625226342,
    "principal_pt_x": 1354.4265245977356,
    "principal_pt_y": 1011.4847310011687,
    "radial_distortion_1": -0.02196117964405394,
    "radial_distortion_2": -0.018959717016668237,
    "radial_distortion_3": 0.001693880829392453,
    "radial_distortion_4": -0.00016807228608000285
  }
}
```

### 3. **IMU Data Extraction**
- **Script**: `scripts_slam_pipeline/01_extract_gopro_imu.py`
- **Docker Image**: `chicheng/openicc:latest` (OpenImuCameraCalibrator)
- **Extraction Tool**: Node.js script from [OpenImuCameraCalibrator](https://github.com/urbste/OpenImuCameraCalibrator/)
- **Data Format**: JSON with accelerometer (ACCL) and gyroscope (GYRO) data

**IMU Data Structure**:
```json
{
  "1": {
    "streams": {
      "ACCL": {
        "samples": [
          {
            "value": [x, y, z],
            "cts": timestamp_ms,
            "date": "ISO8601_datetime",
            "temperature [°C]": temp
          }
        ]
      },
      "GYRO": {
        "samples": [...]
      }
    }
  }
}
```

### 4. **Video Metadata Requirements**
- **Camera Serial Number**: Must be embedded in QuickTime metadata
  - Field: `QuickTime:CameraSerialNumber`
  - Used for: Multi-camera identification and synchronization
  - Files: scripts_slam_pipeline/00_process_videos.py:65, 06_generate_dataset_plan.py:129,163

- **Timecode**: Must have embedded timecode metadata
  - Field: `timecode` in video stream metadata
  - Field: `creation_time` in video stream metadata
  - Used for: Frame-level temporal synchronization
  - Function: `umi/common/timecode_util.py:stream_get_start_datetime()`
  - Critical for multi-camera sync with microsecond precision

### 5. **SLAM Configuration**
- **SLAM System**: ORB_SLAM3 Monocular-Inertial
- **Docker Image**: Contains pre-configured settings
- **Setting File**: `gopro10_maxlens_fisheye_setting_v1_720.yaml`
- **Location**: scripts_slam_pipeline/02_create_map.py:80
- **Mask Resolution**: 2028 x 2704 (matches video resolution)

## Adapting for Different Cameras

### Option 1: Different GoPro Model (e.g., Hero 11, Hero 12)

**What to change**:
1. **Camera Calibration**:
   - Recalibrate using OpenImuCameraCalibrator
   - Create new intrinsics JSON file
   - Update resolution if different (e.g., 5.3K mode = 2704 x 1520)

2. **SLAM Settings**:
   - May need different ORB_SLAM3 settings file
   - Adjust mask dimensions if resolution changed

3. **Firmware**:
   - Install GoPro Labs firmware for your model
   - Verify IMU data recording works

**What stays the same**:
- IMU extraction (same metadata format across GoPro models with Labs firmware)
- Timing/synchronization (QuickTime metadata structure)
- Camera serial number extraction

### Option 2: Non-GoPro Camera with IMU (e.g., Insta360, DJI Action)

**Major changes required**:

1. **IMU Extraction** (scripts_slam_pipeline/01_extract_gopro_imu.py):
   - Replace Docker image/extraction tool
   - Parse vendor-specific IMU metadata format
   - Convert to same JSON structure expected by SLAM

2. **Camera Intrinsics**:
   - Calibrate your camera using OpenImuCameraCalibrator or similar
   - Ensure fisheye distortion model is compatible with ORB_SLAM3

3. **Metadata Handling**:
   - Ensure camera serial number is available (or add custom identifier)
   - Verify timecode/timestamp metadata exists
   - May need to modify `umi/common/timecode_util.py` for different format

4. **SLAM Configuration**:
   - Create custom ORB_SLAM3 settings file
   - Calibrate IMU-camera extrinsics
   - Adjust for different IMU noise characteristics

### Option 3: Camera WITHOUT IMU (e.g., standard DSLR, webcam)

**This is challenging and may reduce tracking quality**:

1. **SLAM System Change**:
   - Switch from Monocular-Inertial to Monocular-only ORB_SLAM3
   - Lose robustness benefits of IMU fusion
   - Lower accuracy in fast motion

2. **Required Changes**:
   - Modify `scripts_slam_pipeline/02_create_map.py` to use monocular mode
   - Remove IMU data extraction step (01_extract_gopro_imu.py)
   - Create monocular-only ORB_SLAM3 config file

3. **Timing/Synchronization**:
   - Must have precise timestamp for each frame
   - Need reliable camera serial number or identifier
   - May need external sync method (e.g., hardware trigger)

## Key Files to Modify

When adapting to a new camera, these files will need changes:

1. **Camera Intrinsics**:
   - `example/calibration/gopro_intrinsics_2_7k.json`
   - Create new file with your camera's calibration

2. **IMU Extraction**:
   - `scripts_slam_pipeline/01_extract_gopro_imu.py`
   - Replace extraction method for your camera

3. **SLAM Configuration**:
   - `scripts_slam_pipeline/02_create_map.py:80`
   - Update settings file path

4. **Mask Generation**:
   - `scripts_slam_pipeline/02_create_map.py:62`
   - Update resolution (2028, 2704) to match your camera

5. **Metadata Extraction** (if format differs):
   - `umi/common/timecode_util.py`
   - Modify for your camera's timestamp format

6. **Video Processing**:
   - `scripts_slam_pipeline/00_process_videos.py`
   - Update if metadata structure differs

## Testing Your Camera Setup

After adapting for a new camera:

1. **Verify IMU Data**: Check that `imu_data.json` is generated with correct structure
2. **Check Intrinsics**: Ensure calibration file has all required fields
3. **Test SLAM**: Run mapping on a short test video
4. **Verify Timing**: Check that timestamps are precise enough (sub-frame accuracy)
5. **Multi-Camera Sync**: Test with multiple cameras if using more than one

## Real-Time Deployment vs. SLAM Processing

**Two different camera use cases in UMI**:

### 1. **Data Collection & SLAM (GoPro Hero)**
- Records video with embedded IMU to MP4 files
- Processed offline with SLAM pipeline
- Requires: GoPro Labs firmware, IMU data, specific intrinsics

### 2. **Real-Time Policy Deployment (GoPro via HDMI)**
- Live video feed via HDMI capture card
- No SLAM processing, just live inference
- Requires: Clean HDMI output (set via QR code)
- Lower requirements: no IMU needed for inference

**Important**: For data collection/training, you need the full GoPro setup with IMU. For deployment only, you can use simpler cameras via HDMI/USB.

## Common Pitfalls

1. **Frame Rate Mismatch**: IMU data must align with video frame rate
2. **Timestamp Precision**: Need microsecond-level precision for multi-camera sync
3. **Distortion Model**: ORB_SLAM3 expects specific fisheye model (Kannala-Brandt)
4. **IMU Frequency**: Low IMU frequency (<100Hz) may reduce tracking quality
5. **Missing Metadata**: QuickTime container required for proper metadata embedding
6. **Confusing SLAM vs Deployment**: Data collection needs full setup; deployment can be simpler

## Recommended Cameras

For easiest adaptation (in order of compatibility):
1. **GoPro Hero 10/11/12** with Labs firmware (minimal changes)
2. **Insta360 cameras** (have IMU, need custom extraction)
3. **DJI Action cameras** (have IMU, need custom extraction)
4. **iPhone** with Filmic Pro (can embed IMU, need custom workflow)
5. **Standard cameras** (no IMU, monocular-only mode, lowest quality)

## Additional Resources

- OpenImuCameraCalibrator: https://github.com/urbste/OpenImuCameraCalibrator/
- ORB_SLAM3: https://github.com/UZ-SLAMLab/ORB_SLAM3
- UMI's ORB_SLAM3 fork: https://github.com/cheng-chi/ORB_SLAM3
- GoPro Labs: https://gopro.com/en/us/info/gopro-labs
