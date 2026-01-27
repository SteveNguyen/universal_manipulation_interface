from typing import Dict, Tuple

import json
import pathlib
import math
import copy
import numpy as np
import cv2
import scipy.interpolate as si

# =================== OpenCV ArUco API Compatibility ===================
# OpenCV 4.7+ changed the ArUco API. This wrapper provides backward compatibility.

def _detect_aruco_markers_compat(image, dictionary, parameters):
    """Compatibility wrapper for ArUco marker detection across OpenCV versions."""
    major, minor = map(int, cv2.__version__.split('.')[:2])

    if major == 4 and minor >= 7:
        # New API (OpenCV 4.7+)
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(image)
    else:
        # Old API (OpenCV 4.6 and earlier)
        corners, ids, rejected = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)

    return corners, ids, rejected

def _estimate_pose_single_markers_compat(corners, marker_size, camera_matrix, dist_coeffs):
    """Compatibility wrapper for ArUco pose estimation across OpenCV versions."""
    major, minor = map(int, cv2.__version__.split('.')[:2])

    if major == 4 and minor >= 7:
        # New API (OpenCV 4.7+) - estimatePoseSingleMarkers is removed
        # Use solvePnP directly
        marker_points = np.array([
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0]
        ], dtype=np.float32)

        rvecs = []
        tvecs = []
        for corner in corners:
            _, rvec, tvec = cv2.solvePnP(
                marker_points, corner, camera_matrix, dist_coeffs, False, cv2.SOLVEPNP_IPPE_SQUARE
            )
            rvecs.append(rvec)
            tvecs.append(tvec)

        return np.array(rvecs), np.array(tvecs), None
    else:
        # Old API
        return cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

# =================== intrinsics ===================

def parse_fisheye_intrinsics(json_data: dict) -> Dict[str, np.ndarray]:
    """
    Reads camera intrinsics from OpenCameraImuCalibration to opencv format.
    Example:
    {
        "final_reproj_error": 0.17053819312281043,
        "fps": 60.0,
        "image_height": 1080,
        "image_width": 1920,
        "intrinsic_type": "FISHEYE",
        "intrinsics": {
            "aspect_ratio": 1.0026582765352035,
            "focal_length": 420.56809123853304,
            "principal_pt_x": 959.857586309181,
            "principal_pt_y": 542.8155851051391,
            "radial_distortion_1": -0.011968137016185161,
            "radial_distortion_2": -0.03929790706019372,
            "radial_distortion_3": 0.018577224235396064,
            "radial_distortion_4": -0.005075629959840777,
            "skew": 0.0
        },
        "nr_calib_images": 129,
        "stabelized": false
    }
    """
    assert json_data['intrinsic_type'] == 'FISHEYE'
    intr_data = json_data['intrinsics']
    
    # img size
    h = json_data['image_height']
    w = json_data['image_width']

    # pinhole parameters
    f = intr_data['focal_length']
    px = intr_data['principal_pt_x']
    py = intr_data['principal_pt_y']
    
    # Kannala-Brandt non-linear parameters for distortion
    kb8 = [
        intr_data['radial_distortion_1'],
        intr_data['radial_distortion_2'],
        intr_data['radial_distortion_3'],
        intr_data['radial_distortion_4']
    ]

    opencv_intr_dict = {
        'DIM': np.array([w, h], dtype=np.int64),
        'K': np.array([
            [f, 0, px],
            [0, f, py],
            [0, 0, 1]
        ], dtype=np.float64),
        'D': np.array([kb8]).T
    }
    return opencv_intr_dict


def convert_fisheye_intrinsics_resolution(
        opencv_intr_dict: Dict[str, np.ndarray], 
        target_resolution: Tuple[int, int]
        ) -> Dict[str, np.ndarray]:
    """
    Convert fisheye intrinsics parameter to a different resolution,
    assuming that images are not cropped in the vertical dimension,
    and only symmetrically cropped/padded in horizontal dimension.
    """
    iw, ih = opencv_intr_dict['DIM']
    iK = opencv_intr_dict['K']
    ifx = iK[0,0]
    ify = iK[1,1]
    ipx = iK[0,2]
    ipy = iK[1,2]

    ow, oh = target_resolution
    ofx = ifx / ih * oh
    ofy = ify / ih * oh
    opx = (ipx - (iw / 2)) / ih * oh + (ow / 2)
    opy = ipy / ih * oh
    oK = np.array([
        [ofx, 0, opx],
        [0, ofy, opy],
        [0, 0, 1]
    ], dtype=np.float64)

    out_intr_dict = copy.deepcopy(opencv_intr_dict)
    out_intr_dict['DIM'] = np.array([ow, oh], dtype=np.int64)
    out_intr_dict['K'] = oK
    return out_intr_dict


class FisheyeRectConverter:
    def __init__(self, K, D, DIM, out_size, out_fov):
        out_size = np.array(out_size)
        # vertical fov
        out_f = (out_size[1] / 2) / np.tan(out_fov/180*np.pi/2)
        out_K = np.array([
            [out_f, 0, out_size[0]/2],
            [0, out_f, out_size[1]/2],
            [0, 0, 1]
        ], dtype=np.float32)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), out_K, out_size, cv2.CV_16SC2)

        self.map1 = map1
        self.map2 = map2
    
    def forward(self, img):
        rect_img = cv2.remap(img, 
            self.map1, self.map2,
            interpolation=cv2.INTER_AREA, 
            borderMode=cv2.BORDER_CONSTANT)
        return rect_img


# ================= ArUcO tag =====================
def parse_aruco_config(aruco_config_dict: dict):
    """
    example:
    aruco_dict:
        predefined: DICT_4X4_50
    marker_size_map: # all unit in meters
        default: 0.15
        12: 0.2
    """
    aruco_dict = get_aruco_dict(**aruco_config_dict['aruco_dict'])

    n_markers = len(aruco_dict.bytesList)
    marker_size_map = aruco_config_dict['marker_size_map']
    default_size = marker_size_map.get('default', None)
    
    out_marker_size_map = dict()
    for marker_id in range(n_markers):
        size = default_size
        if marker_id in marker_size_map:
            size = marker_size_map[marker_id]
        out_marker_size_map[marker_id] = size
    
    result = {
        'aruco_dict': aruco_dict,
        'marker_size_map': out_marker_size_map
    }
    return result


def get_aruco_dict(predefined:str
                   ) -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, predefined))

def detect_localize_aruco_tags(
        img: np.ndarray, 
        aruco_dict: cv2.aruco.Dictionary, 
        marker_size_map: Dict[int, float], 
        fisheye_intr_dict: Dict[str, np.ndarray], 
        refine_subpix: bool=True):
    K = fisheye_intr_dict['K']
    D = fisheye_intr_dict['D']
    param = cv2.aruco.DetectorParameters()
    if refine_subpix:
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, rejectedImgPoints = _detect_aruco_markers_compat(
        image=img, dictionary=aruco_dict, parameters=param)
    if len(corners) == 0:
        return dict()

    tag_dict = dict()
    for this_id, this_corners in zip(ids, corners):
        this_id = int(this_id[0])
        if this_id not in marker_size_map:
            continue

        marker_size_m = marker_size_map[this_id]
        undistorted = cv2.fisheye.undistortPoints(this_corners, K, D, P=K)
        rvec, tvec, markerPoints = _estimate_pose_single_markers_compat(
            undistorted, marker_size_m, K, np.zeros((1,5)))
        tag_dict[this_id] = {
            'rvec': rvec.squeeze() if rvec is not None else rvec,
            'tvec': tvec.squeeze() if tvec is not None else tvec,
            'corners': this_corners.squeeze()
        }
    return tag_dict

def get_charuco_board(
        aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100), 
        tag_id_offset=50,
        grid_size=(8, 5), square_length_mm=50, tag_length_mm=30):
    
    aruco_dict = cv2.aruco.Dictionary(
        aruco_dict.bytesList[tag_id_offset:], 
        aruco_dict.markerSize)
    board = cv2.aruco.CharucoBoard(
        size=grid_size,
        squareLength=square_length_mm/1000,
        markerLength=tag_length_mm/1000,
        dictionary=aruco_dict)
    return board

def draw_charuco_board(board, dpi=300, padding_mm=15):
    grid_size = np.array(board.getChessboardSize())
    square_length_mm = board.getSquareLength() * 1000

    mm_per_inch = 25.4
    board_size_pixel = (grid_size * square_length_mm + padding_mm * 2) / mm_per_inch * dpi
    board_size_pixel = board_size_pixel.round().astype(np.int64)
    padding_pixel = int(padding_mm / mm_per_inch * dpi)
    board_img = board.generateImage(outSize=board_size_pixel, marginSize=padding_pixel)
    return board_img

def get_gripper_width(tag_dict, left_id, right_id, nominal_z=0.072, z_tolerance=0.008):
    zmax = nominal_z + z_tolerance
    zmin = nominal_z - z_tolerance

    left_x = None
    if left_id in tag_dict:
        tvec = tag_dict[left_id]['tvec']
        # check if depth is reasonable (to filter outliers)
        if zmin < tvec[-1] < zmax:
            left_x = tvec[0]

    right_x = None
    if right_id in tag_dict:
        tvec = tag_dict[right_id]['tvec']
        if zmin < tvec[-1] < zmax:
            right_x = tvec[0]

    width = None
    if (left_x is not None) and (right_x is not None):
        width = right_x - left_x
    elif left_x is not None:
        width = abs(left_x) * 2
    elif right_x is not None:
        width = abs(right_x) * 2
    return width


# =========== image mask ====================
# Reference resolution for GoPro 9/10/11 mask polygons (defined at 2704x2028)
GOPRO_MASK_REFERENCE_RESOLUTION = (2028, 2704)  # (height, width)


def canonical_to_pixel_coords(coords, img_shape):
    """Convert canonical coordinates to pixel coordinates.

    Args:
        coords: Canonical coordinates (centered, normalized by height)
        img_shape: Image shape as (height, width)

    Returns:
        Pixel coordinates array
    """
    if img_shape is None:
        raise ValueError("img_shape must be provided")
    pts = np.asarray(coords) * img_shape[0] + np.array(img_shape[::-1]) * 0.5
    return pts


def pixel_coords_to_canonical(pts, img_shape):
    """Convert pixel coordinates to canonical coordinates.

    Args:
        pts: Pixel coordinates
        img_shape: Image shape as (height, width) - the reference resolution for these points

    Returns:
        Canonical coordinates (centered, normalized by height)
    """
    if img_shape is None:
        raise ValueError("img_shape must be provided")
    coords = (np.asarray(pts) - np.array(img_shape[::-1]) * 0.5) / img_shape[0]
    return coords

def draw_canonical_polygon(img: np.ndarray, coords: np.ndarray, color: tuple):
    pts = canonical_to_pixel_coords(coords, img.shape[:2])
    pts = np.round(pts).astype(np.int32)
    cv2.fillPoly(img, pts, color=color)
    return img

def get_mirror_canonical_polygon():
    """Get mirror polygons in canonical coordinates (GoPro 9/10/11).

    Points are defined at reference resolution and converted to canonical coords.
    """
    left_pts = [
        [540, 1700],
        [680, 1450],
        [590, 1070],
        [290, 1130],
        [290, 1770],
        [550, 1770]
    ]
    # Convert from pixel coords at reference resolution to canonical coords
    left_coords = pixel_coords_to_canonical(left_pts, GOPRO_MASK_REFERENCE_RESOLUTION)
    right_coords = left_coords.copy()
    right_coords[:,0] *= -1
    coords = np.stack([left_coords, right_coords])
    return coords


def get_mirror_crop_slices(img_shape=(1080,1920), left=True):
    """Get slice indices for cropping mirror region.

    Points are defined at reference resolution, converted to canonical coords,
    then back to pixel coords at the target img_shape.
    """
    left_pts = [
        [290, 1120],
        [650, 1480]
    ]
    # Convert from pixel coords at reference resolution to canonical coords
    left_coords = pixel_coords_to_canonical(left_pts, GOPRO_MASK_REFERENCE_RESOLUTION)
    if not left:
        left_coords[:,0] *= -1
    left_pts = canonical_to_pixel_coords(left_coords, img_shape=img_shape)
    left_pts = np.round(left_pts).astype(np.int32)
    slices = (
        slice(np.min(left_pts[:,1]), np.max(left_pts[:,1])), 
        slice(np.min(left_pts[:,0]), np.max(left_pts[:,0]))
    )
    return slices


def get_gripper_canonical_polygon():
    """Get gripper polygons in canonical coordinates (GoPro 9/10/11).

    Points are defined at reference resolution and converted to canonical coords.
    """
    left_pts = [
        [1352, 1730],
        [1100, 1700],
        [650, 1500],
        [0, 1350],
        [0, 2028],
        [1352, 2704]
    ]
    # Convert from pixel coords at reference resolution to canonical coords
    left_coords = pixel_coords_to_canonical(left_pts, GOPRO_MASK_REFERENCE_RESOLUTION)
    right_coords = left_coords.copy()
    right_coords[:,0] *= -1
    coords = np.stack([left_coords, right_coords])
    return coords

def get_finger_canonical_polygon(height=0.37, top_width=0.25, bottom_width=1.4):
    """Get finger polygon in canonical coordinates (GoPro 9/10/11).

    Uses parametric definition based on image aspect ratio.
    """
    # Reference resolution for aspect ratio calculation
    img_h, img_w = GOPRO_MASK_REFERENCE_RESOLUTION

    # calculate coordinates
    top_y = 1. - height
    bottom_y = 1.
    width = img_w / img_h
    middle_x = width / 2.
    top_left_x = middle_x - top_width / 2.
    top_right_x = middle_x + top_width / 2.
    bottom_left_x = middle_x - bottom_width / 2.
    bottom_right_x = middle_x + bottom_width / 2.

    top_y *= img_h
    bottom_y *= img_h
    top_left_x *= img_h
    top_right_x *= img_h
    bottom_left_x *= img_h
    bottom_right_x *= img_h

    # create polygon points for opencv API
    points = [[
        [bottom_left_x, bottom_y],
        [top_left_x, top_y],
        [top_right_x, top_y],
        [bottom_right_x, bottom_y]
    ]]
    coords = pixel_coords_to_canonical(points, img_shape=GOPRO_MASK_REFERENCE_RESOLUTION)
    return coords

def draw_predefined_mask(img, color=(0,0,0), mirror=True, gripper=True, finger=True, use_aa=False):
    all_coords = list()
    if mirror:
        all_coords.extend(get_mirror_canonical_polygon())
    if gripper:
        all_coords.extend(get_gripper_canonical_polygon())
    if finger:
        all_coords.extend(get_finger_canonical_polygon())

    for coords in all_coords:
        pts = canonical_to_pixel_coords(coords, img.shape[:2])
        pts = np.round(pts).astype(np.int32)
        flag = cv2.LINE_AA if use_aa else cv2.LINE_8
        cv2.fillPoly(img,[pts], color=color, lineType=flag)
    return img


# =========== Hero 13 mask ====================
# Hero 13 has different mirror positions and finger/gripper geometry
# Reference resolution for Hero 13 mask polygons (2704x2028)
HERO13_MASK_REFERENCE_RESOLUTION = (2028, 2704)  # (height, width)
HERO13_MASK_REFERENCE_WIDTH = 2704
HERO13_MASK_REFERENCE_HEIGHT = 2028


def get_mirror_polygon_hero13():
    """Get Hero 13 mirror polygons (left and right).

    Returns pixel coordinates at reference resolution (2704x2028).
    Caller should scale to actual video resolution.
    """
    ref_w = HERO13_MASK_REFERENCE_WIDTH
    ref_h = HERO13_MASK_REFERENCE_HEIGHT

    # Left mirror polygon (pixel coords at reference resolution)
    left_mirror_pts = np.array([
        [0, 1150],
        [230, 1100],
        [300, 1200],
        [390, 1600],
        [0, ref_h],
    ], dtype=np.int32)

    # Right mirror is horizontally mirrored
    right_mirror_pts = left_mirror_pts.copy()
    right_mirror_pts[:, 0] = ref_w - right_mirror_pts[:, 0]

    return [left_mirror_pts, right_mirror_pts]


def get_finger_polygon_hero13():
    """Get Hero 13 finger polygon (fingers only, not gripper body).

    Returns pixel coordinates at reference resolution (2704x2028).
    Caller should scale to actual video resolution.
    """
    ref_w = HERO13_MASK_REFERENCE_WIDTH
    ref_h = HERO13_MASK_REFERENCE_HEIGHT

    finger_pts = np.array([
        [390, 1700],
        [910, 1320],
        [ref_w-910, 1320],
        [ref_w-390, 1700],
        [ref_w-400, 1750],
        [ref_w//2, 1920],
        [400, 1750],
    ], dtype=np.int32)

    return [finger_pts]


def get_gripper_body_polygon_hero13():
    """Get Hero 13 gripper body polygon (mechanism only, not mirrors or fingers).

    This is used for training data masking - we want to hide the gripper
    mechanism but keep mirrors (useful viewpoints) and fingers (show state) visible.
    Returns pixel coordinates at reference resolution (2704x2028).
    """
    ref_w = HERO13_MASK_REFERENCE_WIDTH
    ref_h = HERO13_MASK_REFERENCE_HEIGHT

    # Contour that follows the gripper body, leaving mirrors and fingers visible
    gripper_body_pts = np.array([
        [0, ref_h],        # bottom-left
        [350, 1600],
        [400, 1750],
        [ref_w//2, 1920],  # center
        [ref_w-400, 1750],
        [ref_w-350, 1600],
        [ref_w, ref_h],    # bottom-right
    ], dtype=np.int32)

    return [gripper_body_pts]


def draw_predefined_mask_hero13(img, color=(0,0,0), mirror=True, gripper=False, finger=True, use_aa=False):
    """Draw predefined mask for Hero 13 camera.

    Polygons are defined at reference resolution (2704x2028) and automatically
    scaled to match the actual image resolution.

    Args:
        img: Image to draw mask on (any resolution - will be scaled)
        color: Color to fill masked regions
        mirror: Whether to mask mirror regions (default True for SLAM, False for training)
        gripper: Whether to mask gripper body/mechanism (default False for SLAM, True for training)
        finger: Whether to mask finger region (default True for SLAM, False for training)
        use_aa: Whether to use anti-aliasing
    """
    img_h, img_w = img.shape[:2]
    reference_h, reference_w = HERO13_MASK_REFERENCE_RESOLUTION

    # Scale factor if image is different resolution
    scale_x = img_w / reference_w
    scale_y = img_h / reference_h

    all_polygons = []
    if mirror:
        all_polygons.extend(get_mirror_polygon_hero13())
    if gripper:
        all_polygons.extend(get_gripper_body_polygon_hero13())
    if finger:
        all_polygons.extend(get_finger_polygon_hero13())

    flag = cv2.LINE_AA if use_aa else cv2.LINE_8

    for pts in all_polygons:
        # Scale coordinates if needed
        scaled_pts = pts.copy().astype(np.float64)
        scaled_pts[:, 0] *= scale_x
        scaled_pts[:, 1] *= scale_y
        scaled_pts = np.round(scaled_pts).astype(np.int32)
        cv2.fillPoly(img, [scaled_pts], color=color, lineType=flag)

    return img

def get_gripper_with_finger_mask(img, height=0.37, top_width=0.25, bottom_width=1.4, color=(0,0,0)):
    # image size
    img_h = img.shape[0]
    img_w = img.shape[1]

    # calculate coordinates
    top_y = 1. - height
    bottom_y = 1.
    width = img_w / img_h
    middle_x = width / 2.
    top_left_x = middle_x - top_width / 2.
    top_right_x = middle_x + top_width / 2.
    bottom_left_x = middle_x - bottom_width / 2.
    bottom_right_x = middle_x + bottom_width / 2.

    top_y *= img_h
    bottom_y *= img_h
    top_left_x *= img_h
    top_right_x *= img_h
    bottom_left_x *= img_h
    bottom_right_x *= img_h

    # create polygon points for opencv API
    points = np.array([[
        [bottom_left_x, bottom_y],
        [top_left_x, top_y],
        [top_right_x, top_y],
        [bottom_right_x, bottom_y]
    ]], dtype=np.int32)

    img = cv2.fillPoly(img, points, color=color, lineType=cv2.LINE_AA)
    return img

def inpaint_tag(img, corners, tag_scale=1.4, n_samples=16):
    # scale corners with respect to geometric center
    center = np.mean(corners, axis=0)
    scaled_corners = tag_scale * (corners - center) + center
    
    # sample pixels on the boundary to obtain median color
    sample_points = si.interp1d(
        [0,1,2,3,4], list(scaled_corners) + [scaled_corners[0]], 
        axis=0)(np.linspace(0,4,n_samples)).astype(np.int32)
    sample_colors = img[
        np.clip(sample_points[:,1], 0, img.shape[0]-1), 
        np.clip(sample_points[:,0], 0, img.shape[1]-1)
    ]
    median_color = np.median(sample_colors, axis=0).astype(img.dtype)
    
    # draw tag with median color
    img = cv2.fillPoly(
        img, scaled_corners[None,...].astype(np.int32), 
        color=median_color.tolist())
    return img

# =========== other utils ====================
def get_image_transform(in_res, out_res, crop_ratio:float = 1.0, bgr_to_rgb: bool=False, no_crop: bool=False):
    """
    Create an image transform function.

    Args:
        in_res: Input resolution (width, height)
        out_res: Output resolution (width, height)
        crop_ratio: Crop ratio for center crop (default 1.0)
        bgr_to_rgb: Convert BGR to RGB
        no_crop: If True, skip cropping and just resize (useful for Hero 13 which has
                 no fisheye black borders, so all pixels are valuable)
    """
    iw, ih = in_res
    ow, oh = out_res
    interp_method = cv2.INTER_AREA

    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    if no_crop:
        # Hero 13: Just resize, no cropping (preserves full field of view including mirrors)
        def transform(img: np.ndarray):
            assert img.shape == ((ih, iw, 3))
            if bgr_to_rgb:
                img = img[:, :, c_slice]
            img = cv2.resize(img, out_res, interpolation=interp_method)
            return img
    else:
        # GoPro 9/10/11: Center crop then resize (removes fisheye black borders)
        ch = round(ih * crop_ratio)
        cw = round(ih * crop_ratio / oh * ow)

        w_slice_start = (iw - cw) // 2
        w_slice = slice(w_slice_start, w_slice_start + cw)
        h_slice_start = (ih - ch) // 2
        h_slice = slice(h_slice_start, h_slice_start + ch)

        def transform(img: np.ndarray):
            assert img.shape == ((ih, iw, 3))
            # crop
            img = img[h_slice, w_slice, c_slice]
            # resize
            img = cv2.resize(img, out_res, interpolation=interp_method)
            return img

    return transform