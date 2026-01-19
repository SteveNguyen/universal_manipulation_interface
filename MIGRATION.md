# Migration from Conda to UV

This document describes the changes made to modernize the UMI project from conda to uv for package management.

## What Changed

### Package Management
- **Before**: Used conda/mamba with `conda_environment.yaml`
- **After**: Uses uv with `pyproject.toml`

### Installation
- **Before**:
  ```bash
  mamba env create -f conda_environment.yaml
  conda activate umi
  ```

- **After**:
  ```bash
  uv venv --python 3.9
  source .venv/bin/activate
  uv pip install --index-strategy unsafe-best-match -e .
  ```

## Dependency Changes

The following packages were updated for compatibility with modern build tools:

1. **gym**: Updated from `0.21.0` to `0.23.1`
   - Reason: gym 0.21.0 has build issues with modern setuptools
   - Impact: Minor API changes, but should be compatible with existing code

2. **av (PyAV)**: Updated from `10.0.*` to `>=10.0.0`
   - Reason: av 10.0 has compilation issues with modern Cython/FFmpeg
   - Impact: Allows installation of latest version (15.x) with better compatibility

3. **opencv-python**: Kept at `4.7.*` with compatibility wrapper
   - Reason: OpenCV 4.7 introduced breaking changes to the ArUco API
   - Impact: Added compatibility wrapper in `umi/common/cv_util.py` to support the new API
   - The old `cv2.aruco.detectMarkers()` was replaced with `ArucoDetector` class
   - The old `cv2.aruco.estimatePoseSingleMarkers()` was removed, now using `cv2.solvePnP()`

3. **python-lmdb**: Renamed to `lmdb`
   - Reason: Different package names between conda and PyPI
   - Impact: No API changes, just package naming

4. **Simulation dependencies**: Made optional
   - **robomimic**, **free-mujoco-py**, **robosuite** are now optional
   - Reason: These packages have complex build requirements (MuJoCo, OpenGL, CMake)
   - Install separately if needed:
     ```bash
     uv pip install robomimic free-mujoco-py
     uv pip install robosuite@https://github.com/cheng-chi/robosuite/archive/3f2c3116f1c0a1916e94bfa28da4d055927d1ab3.tar.gz
     ```

## PyTorch Installation

PyTorch with CUDA 12.1 support is automatically installed from the PyTorch index. The `--index-strategy unsafe-best-match` flag is required to properly resolve packages from both PyPI and the PyTorch index.

## System Dependencies

The following system dependencies are required:
```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libimage-exiftool-perl
```

**Note**: `libimage-exiftool-perl` provides the `exiftool` binary, which was previously included in the conda environment but must be installed separately when using uv.

For SpaceMouse support:
```bash
sudo apt install libspnav-dev spacenavd
sudo systemctl start spacenavd
```

## Verification

After installation, verify the setup:
```bash
python -c "import umi; import diffusion_policy; import torch; print('✓ Installation successful')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Notes

- The conda environment file (`conda_environment.yaml`) has been preserved for reference
- All core functionality is maintained with the new dependency versions
- Simulation environments (robomimic/robosuite) may require additional setup
- Python 3.9 is still required (same as original conda environment)
- **Important**: `exiftool` must be installed as a system package (`libimage-exiftool-perl`) - it cannot be installed via pip/uv

## Troubleshooting

If you encounter issues:

1. **PyTorch CUDA not available**: Ensure you have NVIDIA drivers installed and use the `--index-strategy unsafe-best-match` flag

2. **Simulation packages fail to build**: These are optional. The core UMI pipeline works without them. Install separately if needed.

3. **Import errors**: Make sure you activated the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Reverting to Conda

If you need to use the conda environment, the original `conda_environment.yaml` file is still available:
```bash
mamba env create -f conda_environment.yaml
conda activate umi
```
