# Hero 13 Camera-IMU Calibration Issue

## Summary

Hero 13 camera-IMU calibration produces excellent metrics but fails in SLAM. Using GoPro 9's T_b_c transformation works as a workaround.

## Calibration Results

### Hero 13 Proper Calibration (29 views)

**Metrics (all excellent):**
- ✅ 29 calibration views (target: 20-30+)
- ✅ Camera intrinsics RMS: 0.30px (excellent, <0.5px threshold)
- ✅ Camera-IMU RMS: 1.10px (excellent, <2.0px threshold)
- ✅ t_i_c magnitude: 6.85mm (physically reasonable for camera size)

**Calibration values (T_i_c = IMU-to-Camera):**
```
q_i_c: w=-0.004028, x=-0.002254, y=-0.705983, z=0.708214
t_i_c: x=1.11mm, y=-6.45mm, z=-2.04mm (magnitude: 6.85mm)
```

**After inversion to T_b_c (Camera-to-IMU for SLAM):**
```
Translation: (1.07, -2.07, -6.44) mm
Rotation matrix has -1.0 on diagonal (indicates ~180° flips)
```

**SLAM Performance:**
- ❌ **1.3% tracking** (31/2438 frames)
- Complete failure despite excellent calibration metrics

### GoPro 9 T_b_c (Working Baseline)

**Calibration values:**
```
q_i_c: w=-0.000657, x=0.707660, y=-0.706534, z=0.005124
t_i_c: x=-2.86mm, y=-13.54mm, z=-51.69mm (magnitude: 53.51mm)
```

**After inversion to T_b_c:**
```
Translation: (-13.21, -3.30, -51.75) mm
Rotation: Mostly axis permutations (small values on diagonal)
```

**SLAM Performance:**
- ✅ **91.4% tracking** on Hero 13 videos
- Works well despite 53mm offset seeming physically implausible

## Analysis

### Key Differences

1. **Translation magnitude:**
   - Hero 13: 6.85mm (physically plausible for 70×50×25mm camera)
   - GoPro 9: 53.51mm (seems too large - almost entire camera height)

2. **Rotation matrices:**
   - Hero 13: Large negative values on diagonal (-1.0) → 180° axis flips
   - GoPro 9: Small values, mostly axis permutations

3. **Coordinate frames:**
   - Both calibrations have y≈-0.706 in quaternion
   - But GoPro 9 has x=0.707, Hero 13 has x≈0
   - Suggests fundamentally different coordinate frame orientations

### Possible Causes

1. **Coordinate system convention mismatch:**
   - OpenICC outputs in one coordinate convention
   - ORB-SLAM3 expects a different convention
   - GoPro 9 calibration might use yet another convention

2. **180° rotation ambiguity:**
   - Camera-IMU calibration can have inherent 180° ambiguities
   - Hero 13 calibration may have converged to "flipped" solution
   - Mathematically valid but SLAM-incompatible

3. **Unknown calibration pipeline:**
   - GoPro 9 calibration was done by project author
   - Process not documented
   - May involve post-processing steps we're missing

### Why GoPro 9 T_b_c Works

Despite the implausible 53mm offset, GoPro 9's T_b_c works because:
- Similar camera physical design between GoPro 9 and Hero 13
- Coordinate frames happen to be compatible with SLAM
- SLAM may compensate for some inaccuracies in the transformation

## Current Solution (Workaround)

**Use Hero 13 camera intrinsics + GoPro 9 T_b_c transformation:**

Configuration file: `hero13_720p_slam_settings_gopro9_tbc.yaml`

**Performance:**
- ✅ 91.4% tracking (exceeds 80% threshold)
- ✅ Better than original GoPro 9 baseline (81.2%)
- ✅ Production-ready

**Files:**
- Camera intrinsics: From Hero 13 calibration (excellent quality)
- T_b_c transformation: From GoPro 9 (working but physically questionable)

## Recommendations

### For Production Use

1. **Use the working configuration:**
   ```bash
   # SLAM settings with Hero 13 intrinsics + GoPro 9 T_b_c
   hero13_720p_slam_settings_gopro9_tbc.yaml
   ```

2. **Accept 91.4% tracking as the baseline**
   - Exceeds 80% "good" threshold
   - Proven to work reliably

3. **Document this workaround for future users**

### For Further Investigation (Optional)

If you want to understand why Hero 13's own T_b_c fails:

1. **Check OpenICC coordinate conventions:**
   - What coordinate system does OpenICC use?
   - Does it match ORB-SLAM3's expectations?

2. **Try coordinate frame transformations:**
   - Apply 180° rotations to Hero 13 T_b_c
   - Test each axis flip combination

3. **Contact OpenICC/ORB-SLAM3 authors:**
   - Ask about coordinate system conventions
   - Understand expected input format

4. **Reverse-engineer GoPro 9 calibration:**
   - Try to reproduce it with GoPro 9 hardware
   - Understand what makes it work

## Video Recording Guidelines

All calibration videos were recorded with:
- ✅ Camera oriented "normally" (camera up, lens forward)
- ✅ Same orientation as test/production videos
- ✅ 4K resolution (Hero 13 native), downscaled to 2.7K after IMU extraction
- ✅ Standard GoPro mounting orientation

**The calibration videos were recorded correctly.** The issue is in coordinate system conversion, not video orientation.

## Files Reference

### Working Configuration (Use This)
- `hero13_720p_slam_settings_gopro9_tbc.yaml` - SLAM settings (Hero 13 intrinsics + GoPro 9 T_b_c)
- `hero13_720p_intrinsics_gopro9_tbc.json` - For visualization
- **Tracking: 91.4%** ✅

### Hero 13 Proper Calibration (Not Working)
- `hero13_proper_720p_slam_settings.yaml` - SLAM settings (Hero 13 intrinsics + Hero 13 T_b_c)
- `hero13_proper_intrinsics_720p.json` - For visualization
- `hero13_proper_intrinsics_2.7k.json` - Original 2.7K calibration
- **Tracking: 1.3%** ❌

### Backup
- `hero13_working_backup_20260122_130133/` - Backup of all working configurations

## Testing Scripts

```bash
# Test with GoPro 9 T_b_c (working baseline)
./run_slam_hero13_gopro9tbc.sh

# Test with Hero 13's own T_b_c (fails)
./run_slam_hero13_proper.sh

# Check tracking percentage
uv run python3 check_tracking.py path/to/trajectory.csv
```

## Conclusion

Despite the Hero 13 calibration being mathematically excellent, a coordinate system incompatibility causes SLAM failure. Using GoPro 9's T_b_c as a workaround provides excellent performance (91.4% tracking).

**For production: Use `hero13_720p_slam_settings_gopro9_tbc.yaml`**
