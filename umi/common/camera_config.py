"""
Camera configuration module for resolution-agnostic UMI pipeline.

This module provides:
- Camera configuration profiles with resolution and intrinsics info
- Functions for scaling intrinsics between resolutions
- Auto-generation of ORB-SLAM3 settings from intrinsics

Usage:
    from umi.common.camera_config import CAMERA_CONFIGS, scale_intrinsics, get_video_resolution

    config = CAMERA_CONFIGS['hero13']
    intrinsics = scale_intrinsics(source_intrinsics, source_res=(2704, 2028), target_res=(960, 720))
"""

import json
import pathlib
from typing import Dict, Tuple, Optional, Any
import subprocess


# Camera configuration profiles
# Each camera type has its specific resolutions and intrinsics file
CAMERA_CONFIGS = {
    'hero13': {
        'native_resolution': (4000, 3000),      # 4K native (8:6 aspect)
        'slam_input_resolution': (2704, 2028),  # Input to SLAM (after downscale from 4K)
        'slam_resolution': (960, 720),          # ORB-SLAM3 internal processing
        'reference_intrinsics': 'hero13_proper_intrinsics_2.7k.json',
        'reference_intrinsics_resolution': (2704, 2028),
        'mask_type': 'hero13',
        'mask_reference_resolution': (2704, 2028),  # Polygons defined at this resolution
    },
    'gopro9': {
        'native_resolution': (2704, 2028),      # 2.7K native
        'slam_input_resolution': (1920, 1080),  # Input to SLAM
        'slam_resolution': (960, 720),          # ORB-SLAM3 internal processing
        'reference_intrinsics': 'gopro_intrinsics_2_7k.json',
        'reference_intrinsics_resolution': (2704, 2028),
        'mask_type': 'gopro',
        'mask_reference_resolution': (1920, 1080),  # Polygons defined at this resolution
    },
    'gopro10': {
        'native_resolution': (2704, 2028),
        'slam_input_resolution': (1920, 1080),
        'slam_resolution': (960, 720),
        'reference_intrinsics': 'gopro_intrinsics_2_7k.json',
        'reference_intrinsics_resolution': (2704, 2028),
        'mask_type': 'gopro',
        'mask_reference_resolution': (1920, 1080),
    },
    'gopro11': {
        'native_resolution': (2704, 2028),
        'slam_input_resolution': (1920, 1080),
        'slam_resolution': (960, 720),
        'reference_intrinsics': 'gopro_intrinsics_2_7k.json',
        'reference_intrinsics_resolution': (2704, 2028),
        'mask_type': 'gopro',
        'mask_reference_resolution': (1920, 1080),
    },
}


def get_video_resolution(video_path) -> Tuple[int, int]:
    """Get resolution from video file.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height)
    """
    video_path = pathlib.Path(video_path)

    # Try using av (PyAV) first
    try:
        import av
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            return stream.width, stream.height
    except ImportError:
        pass

    # Fall back to ffprobe
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        str(video_path)
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    w, h = map(int, result.stdout.strip().split(','))
    return w, h


def scale_intrinsics(intrinsics: Dict[str, Any],
                     source_res: Tuple[int, int],
                     target_res: Tuple[int, int]) -> Dict[str, Any]:
    """Scale camera intrinsics from source to target resolution.

    Intrinsic parameters (fx, fy, cx, cy) scale linearly with resolution.
    Distortion coefficients (k1-k4) are independent of resolution.

    Args:
        intrinsics: Dict with 'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'
                    or OpenCameraImuCalibration format
        source_res: Source resolution (width, height)
        target_res: Target resolution (width, height)

    Returns:
        Scaled intrinsics dict
    """
    src_w, src_h = source_res
    tgt_w, tgt_h = target_res

    scale_x = tgt_w / src_w
    scale_y = tgt_h / src_h

    # Handle OpenCameraImuCalibration format
    if 'intrinsics' in intrinsics:
        intr = intrinsics['intrinsics']
        f = intr['focal_length']
        aspect = intr.get('aspect_ratio', 1.0)
        return {
            'fx': f * scale_x,
            'fy': f * aspect * scale_y,
            'cx': intr['principal_pt_x'] * scale_x,
            'cy': intr['principal_pt_y'] * scale_y,
            'k1': intr['radial_distortion_1'],
            'k2': intr['radial_distortion_2'],
            'k3': intr['radial_distortion_3'],
            'k4': intr['radial_distortion_4'],
        }

    # Handle flat format
    return {
        'fx': intrinsics['fx'] * scale_x,
        'fy': intrinsics['fy'] * scale_y,
        'cx': intrinsics['cx'] * scale_x,
        'cy': intrinsics['cy'] * scale_y,
        'k1': intrinsics['k1'],
        'k2': intrinsics['k2'],
        'k3': intrinsics['k3'],
        'k4': intrinsics['k4'],
    }


def load_intrinsics(intrinsics_path: pathlib.Path) -> Dict[str, Any]:
    """Load intrinsics from JSON file.

    Supports OpenCameraImuCalibration format and flat format.
    Returns normalized format with fx, fy, cx, cy, k1-k4.
    """
    with open(intrinsics_path) as f:
        data = json.load(f)

    # OpenCameraImuCalibration format
    if 'intrinsics' in data:
        intr = data['intrinsics']
        f = intr['focal_length']
        aspect = intr.get('aspect_ratio', 1.0)
        return {
            'fx': f,
            'fy': f * aspect,
            'cx': intr['principal_pt_x'],
            'cy': intr['principal_pt_y'],
            'k1': intr['radial_distortion_1'],
            'k2': intr['radial_distortion_2'],
            'k3': intr['radial_distortion_3'],
            'k4': intr['radial_distortion_4'],
            'image_width': data.get('image_width'),
            'image_height': data.get('image_height'),
            'fps': data.get('fps', 60),
            # Include IMU-camera transform if available
            't_i_c': data.get('t_i_c'),
            'q_i_c': data.get('q_i_c'),
        }

    # Already flat format
    return data


def generate_slam_settings(intrinsics_path: pathlib.Path,
                           slam_resolution: Tuple[int, int],
                           output_path: pathlib.Path,
                           imu_params: Optional[Dict] = None,
                           camera_imu_transform: Optional[Dict] = None) -> pathlib.Path:
    """Generate ORB-SLAM3 settings YAML from intrinsics JSON.

    Args:
        intrinsics_path: Path to intrinsics JSON file
        slam_resolution: Target resolution for SLAM (width, height)
        output_path: Path to save generated YAML
        imu_params: Optional IMU noise parameters (uses defaults if None)
        camera_imu_transform: Optional camera-IMU transform matrix (4x4)

    Returns:
        Path to generated YAML file
    """
    intrinsics_path = pathlib.Path(intrinsics_path)
    output_path = pathlib.Path(output_path)

    # Load intrinsics
    with open(intrinsics_path) as f:
        raw_data = json.load(f)

    # Get source resolution
    if 'image_width' in raw_data and 'image_height' in raw_data:
        src_w = raw_data['image_width']
        src_h = raw_data['image_height']
    else:
        raise ValueError("Intrinsics file must contain image_width and image_height")

    source_res = (src_w, src_h)
    intrinsics = load_intrinsics(intrinsics_path)

    # Scale to SLAM resolution
    tgt_w, tgt_h = slam_resolution
    scaled = scale_intrinsics(intrinsics, source_res, slam_resolution)

    fps = intrinsics.get('fps', 60)

    # Default IMU parameters (GoPro typical values)
    if imu_params is None:
        imu_params = {
            'NoiseGyro': 0.0015,
            'NoiseAcc': 0.017,
            'GyroWalk': 5.0e-5,
            'AccWalk': 0.0055,
            'Frequency': 200.0,
        }

    # Default camera-IMU transform (identity-ish, needs calibration)
    if camera_imu_transform is None:
        # Try to get from intrinsics file if available
        if intrinsics.get('t_i_c') and intrinsics.get('q_i_c'):
            # Convert quaternion + translation to 4x4 matrix
            t = intrinsics['t_i_c']
            q = intrinsics['q_i_c']
            # This is a simplified conversion - proper implementation would use scipy
            camera_imu_transform = _quat_trans_to_matrix(q, t)
        else:
            # Use GoPro 9 default as fallback (commonly used)
            camera_imu_transform = [
                0.00156717, -0.99997878, 0.00632289, -0.01321271,
                -0.99996531, -0.00161881, -0.00817069, -0.00330095,
                0.00818075, -0.00630987, -0.99994663, -0.05175258,
                0.0, 0.0, 0.0, 1.0
            ]

    # Format transform matrix for YAML
    if isinstance(camera_imu_transform, list) and len(camera_imu_transform) == 16:
        transform_data = camera_imu_transform
    else:
        transform_data = list(camera_imu_transform.flatten()) if hasattr(camera_imu_transform, 'flatten') else camera_imu_transform

    # Generate YAML content
    yaml_content = f"""%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters - Auto-generated from {intrinsics_path.name}
# Source resolution: {src_w}x{src_h} -> SLAM resolution: {tgt_w}x{tgt_h}
#--------------------------------------------------------------------------------------------
File.version: "1.0"
Camera.type: "KannalaBrandt8"

# Camera calibration (scaled from {src_w}x{src_h} to {tgt_w}x{tgt_h})
Camera1.fx: {scaled['fx']:.9f}
Camera1.fy: {scaled['fy']:.9f}
Camera1.cx: {scaled['cx']:.9f}
Camera1.cy: {scaled['cy']:.9f}

Camera1.k1: {scaled['k1']:.15f}
Camera1.k2: {scaled['k2']:.15f}
Camera1.k3: {scaled['k3']:.15f}
Camera1.k4: {scaled['k4']:.15f}

# Camera resolution
Camera.width: {tgt_w}
Camera.height: {tgt_h}

# Camera frames per second
Camera.fps: {fps}

# Color order of the images (0: BGR, 1: RGB)
Camera.RGB: 1

# Transformation from camera to imu (body frame)
IMU.T_b_c1: !!opencv-matrix
    rows: 4
    cols: 4
    dt: f
    data: [{', '.join(f'{v:.8f}' for v in transform_data)}]

# IMU noise parameters
IMU.NoiseGyro: {imu_params['NoiseGyro']}
IMU.NoiseAcc: {imu_params['NoiseAcc']}
IMU.GyroWalk: {imu_params['GyroWalk']}
IMU.AccWalk: {imu_params['AccWalk']}
IMU.Frequency: {imu_params['Frequency']}

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1250

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

System.thFarPoints: 20.0

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -3.5
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(yaml_content)

    return output_path


def _quat_trans_to_matrix(q: Dict, t: Dict) -> list:
    """Convert quaternion and translation to 4x4 transformation matrix.

    Args:
        q: Dict with keys 'w', 'x', 'y', 'z'
        t: Dict with keys 'x', 'y', 'z'

    Returns:
        Flattened 4x4 matrix as list of 16 floats
    """
    qw, qx, qy, qz = q['w'], q['x'], q['y'], q['z']
    tx, ty, tz = t['x'], t['y'], t['z']

    # Rotation matrix from quaternion
    r00 = 1 - 2*(qy*qy + qz*qz)
    r01 = 2*(qx*qy - qz*qw)
    r02 = 2*(qx*qz + qy*qw)
    r10 = 2*(qx*qy + qz*qw)
    r11 = 1 - 2*(qx*qx + qz*qz)
    r12 = 2*(qy*qz - qx*qw)
    r20 = 2*(qx*qz - qy*qw)
    r21 = 2*(qy*qz + qx*qw)
    r22 = 1 - 2*(qx*qx + qy*qy)

    return [
        r00, r01, r02, tx,
        r10, r11, r12, ty,
        r20, r21, r22, tz,
        0.0, 0.0, 0.0, 1.0
    ]


def get_mask_resolution(camera_type: str, video_resolution: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """Get appropriate mask resolution for camera type.

    If video_resolution is provided, returns it (mask should match video).
    Otherwise returns the reference resolution for the camera type.

    Args:
        camera_type: Camera type (e.g., 'hero13', 'gopro9')
        video_resolution: Optional actual video resolution

    Returns:
        Tuple of (width, height) for mask creation
    """
    if video_resolution is not None:
        return video_resolution

    config = CAMERA_CONFIGS.get(camera_type, CAMERA_CONFIGS['gopro9'])
    return config['mask_reference_resolution']


def get_camera_config(camera_type: str) -> Dict[str, Any]:
    """Get camera configuration by type.

    Args:
        camera_type: Camera type string (e.g., 'hero13', 'gopro9')

    Returns:
        Camera configuration dict
    """
    if camera_type not in CAMERA_CONFIGS:
        print(f"Warning: Unknown camera type '{camera_type}', using 'gopro9' defaults")
        return CAMERA_CONFIGS['gopro9']
    return CAMERA_CONFIGS[camera_type]
