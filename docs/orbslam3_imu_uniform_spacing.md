# ORB-SLAM3 IMU Uniform Spacing Requirement

## Summary

ORB-SLAM3 requires uniformly spaced IMU measurements. Non-uniform timestamps (even with correct values) cause tracking failure due to a noise model assumption in the pre-integration code. The workaround is to resample IMU data to a uniform grid before feeding it to SLAM.

## Evidence

With BNO080 IMU data at ~200Hz (median gap 4.81ms, std dev 1.4ms, max gap 16.5ms):

| IMU Data | Tracking |
|----------|----------|
| Raw (non-uniform) | 30% |
| Resampled to uniform 200Hz | 100% |

Scale and maxspan are identical when tracking succeeds — the timestamps are accurate, they just need to be uniform.

## Root Cause

### Noise scaling in `Tracking.cc` (lines 204-211)

```cpp
mImuFreq = settings->imuFrequency();  // from YAML: 200
const float sf = sqrt(mImuFreq);      // sqrt(200) = 14.14
mpImuCalib = new IMU::Calib(Tbc, Ng*sf, Na*sf, Ngw/sf, Naw/sf);
```

The YAML noise parameters are in **spectral density** units (per-sqrt-Hz). ORB-SLAM3 converts them to discrete-time values by multiplying by `sqrt(IMU.Frequency)`. This pre-scaled noise matrix `Nga` is fixed at initialization and used in every pre-integration step.

### Covariance propagation in `ImuTypes.cc` (`IntegrateNewMeasurement`)

```cpp
B.block<3,3>(0,0) = dRi.rightJ * dt;
B.block<3,3>(3,3) = dR * dt;
B.block<3,3>(6,3) = 0.5f * dR * dt * dt;

C = A * C * A.transpose() + B * Nga * B.transpose();
```

The noise contribution per step is `B * Nga * B^T`, which scales as `dt² * Nga`. Since `Nga` was pre-scaled assuming `dt = 1/f_imu`, this is only correct when every step has exactly that duration.

### The math

For uniform spacing (`dt = 1/f`):
- Noise per step: `dt² * Ng² * f = Ng² / f` (correct)
- Sum over `N = f*T` steps: `Ng² * T` (correct continuous-time total)

For non-uniform spacing (`dt` varies):
- A 7ms step (vs expected 5ms): contributes `(7/5)² = 1.96x` too much noise
- A 3ms step: contributes `(3/5)² = 0.36x` too little noise
- Covariance estimate becomes wrong → optimizer gets bad information matrices → tracking fails

### Developer's own doubt

There's a suspicious hardcoded value with a TODO comment (in Spanish):
```cpp
mImuPer = 0.001; //1.0 / (double) mImuFreq;  //TODO: ESTO ESTA BIEN?
// "IS THIS OK?"
```

This hardcodes the IMU period to 1ms regardless of `IMU.Frequency`, used as a tolerance when matching IMU samples to frame boundaries.

## Correct Fix (future work)

Replace the pre-scaled `Nga` with continuous-time spectral density and scale per step:

```cpp
// Current (assumes uniform dt):
// Nga is pre-scaled by IMU.Frequency at init
C = A * C * A^T + B * Nga * B^T;

// Correct (handles any dt):
// Store Nga_continuous = diag(Ng², Na²) without frequency scaling
// Scale per step by 1/dt to convert spectral density to discrete noise
C = A * C * A^T + B * (Nga_continuous / dt) * B^T;
```

For the rotation block, this gives `rightJ² * dt * sigma²` — proportional to `dt` as expected for a continuous-time noise process, regardless of step size.

The same fix should be applied to the bias random walk terms (`Ngw`, `Naw`), which are currently scaled by `1/sqrt(f)` and should instead be scaled by `dt` per step.

### Files to modify

- `/ORB_SLAM3/src/Tracking.cc` — Remove `sqrt(mImuFreq)` scaling at init
- `/ORB_SLAM3/src/ImuTypes.cc` — `IntegrateNewMeasurement()`: divide `Nga` by `dt` per step
- `/ORB_SLAM3/include/ImuTypes.h` — Store continuous-time noise parameters

## Current Workaround

Resample IMU data to uniform `1/IMU.Frequency` spacing (5ms for 200Hz) using linear interpolation before feeding to SLAM. This makes every `dt` match the assumed value, so the noise model is correct.
