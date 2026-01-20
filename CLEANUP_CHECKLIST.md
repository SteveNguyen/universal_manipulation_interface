# Hero 13 Integration - Cleanup Checklist

Complete this checklist before committing Hero 13 integration to repository.

## ✅ Documentation Created

- [x] `HERO13_SETUP.md` - Complete setup guide with calibration procedure
- [x] `HERO13_TECHNICAL_NOTES.md` - Technical investigation findings
- [x] `HERO13_README.md` - Quick reference for files and structure
- [x] `CLEANUP_CHECKLIST.md` - This file

## ✅ Essential Files Ready

- [x] `hero13_720p_intrinsics.json` - Final working calibration
- [x] `create_720p_calibration.py` - Script to regenerate from OpenICC
- [x] `test_hero13_slam.py` - SLAM validation tool
- [x] `cleanup_hero13_dev.sh` - Cleanup script

## 📋 Pre-Commit Tasks

### 1. Review Documentation
- [ ] Read through `HERO13_SETUP.md` for accuracy
- [ ] Verify all file paths in documentation are correct
- [ ] Check that troubleshooting section is complete
- [ ] Ensure all commands are tested and work

### 2. Verify Calibration Files
- [ ] Confirm `hero13_720p_intrinsics.json` has all required fields:
  - [ ] `aspect_ratio`
  - [ ] `focal_length`
  - [ ] `principal_pt_x`, `principal_pt_y`
  - [ ] `radial_distortion_1` through `4`
  - [ ] `skew`
  - [ ] `q_i_c` (quaternion)
  - [ ] `t_i_c` (translation)
  - [ ] `cam_imu_reproj_error`

### 3. Test Scripts Work
- [ ] Run `create_720p_calibration.py` successfully
- [ ] Verify it produces valid `hero13_720p_intrinsics.json`
- [ ] Run `test_hero13_slam.py` with a test video
- [ ] Confirm SLAM test passes (>50 tracked poses)

### 4. Run Cleanup
- [ ] Review `cleanup_hero13_dev.sh` contents
- [ ] Run cleanup script: `./cleanup_hero13_dev.sh`
- [ ] Verify archive was created
- [ ] Check that essential files remain

### 5. Update Main Documentation (if exists)
- [ ] Add Hero 13 to main README if applicable
- [ ] Update any hardware compatibility lists
- [ ] Link to `HERO13_SETUP.md` from main docs

### 6. Prepare for Integration
- [ ] Copy `hero13_720p_intrinsics.json` to `example/calibration/`
- [ ] Update `scripts_slam_pipeline/02_create_map.py` if needed
- [ ] Update `scripts_slam_pipeline/04_detect_aruco.py` if needed
- [ ] Test full pipeline with Hero 13 data

## 🗑️ Files to Remove (via cleanup script)

The `cleanup_hero13_dev.sh` script will archive:

**Test Directories:**
- test_hero13_slam
- test_hero13_slam_v2
- test_hero13_slam_mono
- test_hero13_slam_openicc
- test_hero13_slam_correct_noise
- test_hero13_slam_trimmed_noise
- test_hero13_slam_gopro10_noise
- test_gopro9_baseline

**Intermediate Calibrations:**
- hero13_4k_intrinsics.json
- hero13_4k_intrinsics_openicc.json
- hero13_4k_intrinsics_openicc_v2.json
- hero13_intrinsics.json
- example/calibration/hero13_4k_intrinsics.json

**Experimental Settings:**
- hero13_4k_slam_settings.yaml
- hero13_4k_slam_settings_openicc.yaml
- hero13_4k_slam_settings_openicc_correct_noise.yaml
- hero13_openicc_slam_settings.yaml

**Experimental Scripts:**
- calibrate_hero13_openicc.py
- calibrate_hero13_pinhole.py
- calibrate_hero13_safe.py
- extract_imu_noise.py
- extract_imu_noise_from_camimu_calib.py
- plot_camimu_residuals.py
- plot_static_imu.py
- convert_openicc_to_slam.py
- test_hero13_slam_monocular.py
- test_undistort.py
- create_slam_mask.py
- convert_hero13_to_h264.sh

**Debug Directories:**
- calibration_debug
- calibration_debug_pinhole
- hero13_calib
- hero13_openicc_calib
- openicc_work
- test_hero13
- test_slam_hero13
- test_undistort
- undistort_test
- bak

**Temporary Files:**
- camimu_residuals_plot.png
- static_imu_plot.png
- charuco_board.png
- .*.~undo-tree~
- Dockerfile.openicc
- OPENICC_CALIBRATION_GUIDE.md

## 📦 Files to Keep

**Calibration:**
- hero13_720p_intrinsics.json
- hero13_openicc_dataset/

**Scripts:**
- create_720p_calibration.py
- test_hero13_slam.py
- cleanup_hero13_dev.sh

**Documentation:**
- HERO13_SETUP.md
- HERO13_TECHNICAL_NOTES.md
- HERO13_README.md
- CLEANUP_CHECKLIST.md

**Examples:**
- test_hero13_slam_720p/
- OpenImuCameraCalibrator/

## 🚀 Post-Cleanup Actions

After running cleanup:

1. **Review Archive**
   ```bash
   ls hero13_dev_archive_*/
   ```

2. **Test Clean Repository**
   ```bash
   # Verify essential files exist
   ls hero13_720p_intrinsics.json
   ls create_720p_calibration.py
   ls test_hero13_slam.py

   # Run validation
   python create_720p_calibration.py
   ```

3. **Commit to Git** (if using version control)
   ```bash
   git add hero13_720p_intrinsics.json
   git add create_720p_calibration.py
   git add test_hero13_slam.py
   git add HERO13_*.md
   git add CLEANUP_CHECKLIST.md
   git commit -m "Add GoPro Hero 13 support with 720p calibration"
   ```

4. **Delete Archive** (when confident)
   ```bash
   rm -rf hero13_dev_archive_*
   ```

## 📝 Notes

- Archive is created in case you need to reference old files
- Review archive contents before deleting
- Keep `hero13_openicc_dataset/` - it's the source data
- `test_hero13_slam_720p/` serves as working example
- All removed files can be regenerated if needed

## ✔️ Final Verification

Before declaring cleanup complete:

- [ ] All documentation is accurate and complete
- [ ] Essential files are present and working
- [ ] Test scripts run successfully
- [ ] Archive contains all removed files
- [ ] Repository is clean and organized
- [ ] Integration with main pipeline is documented

## Success Criteria

✅ Repository is clean and professional
✅ Documentation is comprehensive
✅ Hero 13 calibration works reliably
✅ Future users can follow setup guide
✅ No orphaned or unexplained files

**Date Completed:** _______________

**Verified By:** _______________
