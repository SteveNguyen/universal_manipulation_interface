# GoPro Hero 13 Files

Quick reference for Hero 13 integration files.

## Essential Files

### Calibration
- **`hero13_720p_intrinsics.json`** - Final working calibration for SLAM
  - Resolution: 960x720
  - Camera RMS: 0.19 pixels
  - Camera-IMU RMS: 0.65 pixels
  - Copy to `example/calibration/` for production use

### Scripts
- **`create_720p_calibration.py`** - Generate 720p calibration from OpenICC 4K output
  - Input: `hero13_openicc_dataset/cam/cam_calib_*.json` (camera)
  - Input: `hero13_openicc_dataset/cam_imu/cam_imu_calib_result_*.json` (T_cam_imu)
  - Output: `hero13_720p_intrinsics.json`

- **`test_hero13_slam.py`** - Validate calibration with SLAM test
  - Tests if calibration works with ORB-SLAM3
  - Reports tracking quality
  - Useful after re-calibration

### Data
- **`hero13_openicc_dataset/`** - Original OpenICC calibration data
  - `cam/` - Camera-only calibration results (4K FISHEYE)
  - `cam_imu/` - Camera-IMU calibration results (T_cam_imu)
  - `imu_bias/` - Static IMU bias measurements
  - Keep this for reference and to regenerate 720p calibration

- **`test_hero13_slam_720p/`** - Successful SLAM test output (example)
  - 174 tracked poses
  - Demonstrates working configuration
  - Can be used as baseline for comparison

### Documentation
- **`HERO13_SETUP.md`** - Complete setup guide
  - Calibration procedure
  - Integration with UMI pipeline
  - Troubleshooting

- **`HERO13_TECHNICAL_NOTES.md`** - Technical investigation notes
  - Design decisions explained
  - Dead ends investigated
  - Lessons learned

### Cleanup
- **`cleanup_hero13_dev.sh`** - Script to remove development/test files
  - Archives temporary files
  - Keeps essential files
  - Run after reviewing documentation

## Directory Structure (after cleanup)

```
universal_manipulation_interface/
├── hero13_720p_intrinsics.json          # Final calibration
├── create_720p_calibration.py            # Regeneration script
├── test_hero13_slam.py                   # Validation tool
├── HERO13_SETUP.md                       # Setup guide
├── HERO13_TECHNICAL_NOTES.md             # Technical details
├── HERO13_README.md                      # This file
├── cleanup_hero13_dev.sh                 # Cleanup script
│
├── hero13_openicc_dataset/               # OpenICC calibration data
│   ├── cam/                              #   Camera calibration
│   ├── cam_imu/                          #   Camera-IMU calibration
│   └── imu_bias/                         #   IMU bias
│
├── test_hero13_slam_720p/                # Successful test (example)
│   └── demos/mapping/
│       ├── raw_video.mp4
│       ├── imu_data.json
│       ├── camera_trajectory.csv         # 174 poses
│       └── map_atlas.osa
│
├── OpenImuCameraCalibrator/              # Calibration tool (submodule)
│
└── example/calibration/
    └── hero13_720p_intrinsics.json       # Copy here for production
```

## Quick Start

**If starting fresh:**
1. Read `HERO13_SETUP.md` for complete procedure
2. Record calibration videos (see guide)
3. Run OpenICC calibration
4. Generate 720p: `python create_720p_calibration.py`
5. Validate: `python test_hero13_slam.py <video> --intrinsics hero13_720p_intrinsics.json`
6. Copy to production: `cp hero13_720p_intrinsics.json example/calibration/`

**If using existing calibration:**
1. Use `hero13_720p_intrinsics.json` directly
2. Copy to `example/calibration/`
3. Update pipeline scripts (see `HERO13_SETUP.md`)

## File Sizes (approximate)

```
hero13_720p_intrinsics.json              1 KB
create_720p_calibration.py               5 KB
test_hero13_slam.py                     15 KB
HERO13_SETUP.md                         20 KB
HERO13_TECHNICAL_NOTES.md               15 KB
hero13_openicc_dataset/                500 MB (videos + results)
test_hero13_slam_720p/                  50 MB (includes video)
OpenImuCameraCalibrator/               500 MB (submodule)
```

## Cleanup Instructions

After reviewing documentation and confirming everything works:

```bash
# Review what will be removed
cat cleanup_hero13_dev.sh

# Run cleanup (creates archive)
./cleanup_hero13_dev.sh

# Later, delete archive if not needed
rm -rf hero13_dev_archive_*
```

This removes:
- All failed/experimental test directories
- Intermediate calibration attempts
- Debug scripts and plots
- Temporary work directories

Keeps:
- Final calibration
- Documentation
- One successful test example
- Original OpenICC dataset

## Support

If SLAM fails after following setup guide:
1. Check `HERO13_SETUP.md` → Troubleshooting section
2. Review `HERO13_TECHNICAL_NOTES.md` → What Actually Worked
3. Compare your test output with `test_hero13_slam_720p/`

Common issues:
- Using 4K settings instead of 720p → SLAM fails
- Missing `aspect_ratio` in calibration file → Script error
- Wrong IMU noise parameters → "scale too small" error
- Video has insufficient motion → Initialization failure

## Version History

**v1.0 - 2025-01-20**
- Initial Hero 13 calibration and integration
- Validated with 174 tracked poses
- Documented complete workflow
- Created cleanup and regeneration tools
