#!/usr/bin/env python3
"""
Visualize the predefined masks for GoPro 9/10/11 and Hero 13.
This helps understand what regions are masked during dataset generation.
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

import cv2
import numpy as np
import click
from pathlib import Path

from umi.common.cv_util import (
    draw_predefined_mask,
    draw_predefined_mask_hero13,
    get_mirror_canonical_polygon,
    get_gripper_canonical_polygon,
    get_finger_canonical_polygon,
    canonical_to_pixel_coords,
    get_mirror_polygon_hero13,
    get_finger_polygon_hero13,
)


def visualize_gopro9_masks(img):
    """Visualize GoPro 9/10/11 masks with different colors."""
    h, w = img.shape[:2]

    # Mirror mask (blue)
    mirror_coords = get_mirror_canonical_polygon()
    for coords in mirror_coords:
        pts = canonical_to_pixel_coords(coords, (h, w))
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(img, [pts], color=(255, 0, 0), lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], True, color=(255, 255, 255), thickness=2)

    # Gripper mask (green)
    gripper_coords = get_gripper_canonical_polygon()
    for coords in gripper_coords:
        pts = canonical_to_pixel_coords(coords, (h, w))
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(img, [pts], color=(0, 255, 0), lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], True, color=(255, 255, 255), thickness=2)

    # Finger mask (red)
    finger_coords = get_finger_canonical_polygon()
    for coords in finger_coords:
        pts = canonical_to_pixel_coords(coords, (h, w))
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(img, [pts], color=(0, 0, 255), lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], True, color=(255, 255, 255), thickness=2)

    # Add legend
    cv2.putText(img, "GoPro 9/10/11 Masks", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "Blue: Mirror", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, "Green: Gripper", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, "Red: Finger", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return img


def visualize_hero13_masks(img):
    """Visualize Hero 13 masks with different colors."""
    h, w = img.shape[:2]
    reference_h, reference_w = 2028, 2704
    scale_x = w / reference_w
    scale_y = h / reference_h

    # Mirror mask (blue)
    mirror_polys = get_mirror_polygon_hero13()
    for pts in mirror_polys:
        scaled_pts = pts.copy().astype(np.float64)
        scaled_pts[:, 0] *= scale_x
        scaled_pts[:, 1] *= scale_y
        scaled_pts = np.round(scaled_pts).astype(np.int32)
        cv2.fillPoly(img, [scaled_pts], color=(255, 0, 0), lineType=cv2.LINE_AA)
        cv2.polylines(img, [scaled_pts], True, color=(255, 255, 255), thickness=2)

    # Finger/Gripper mask (red) - current Hero 13 mask combines both
    finger_polys = get_finger_polygon_hero13()
    for pts in finger_polys:
        scaled_pts = pts.copy().astype(np.float64)
        scaled_pts[:, 0] *= scale_x
        scaled_pts[:, 1] *= scale_y
        scaled_pts = np.round(scaled_pts).astype(np.int32)
        cv2.fillPoly(img, [scaled_pts], color=(0, 0, 255), lineType=cv2.LINE_AA)
        cv2.polylines(img, [scaled_pts], True, color=(255, 255, 255), thickness=2)

    # Add legend
    cv2.putText(img, "Hero 13 Masks", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "Blue: Mirror", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, "Red: Finger+Gripper", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return img


@click.command()
@click.option('--video', '-v', type=click.Path(exists=True), default=None,
              help='Optional video file to overlay masks on a frame')
@click.option('--frame', '-f', type=int, default=100,
              help='Frame number to extract from video')
@click.option('--output', '-o', type=click.Path(), default=None,
              help='Output path for mask visualization')
def main(video, frame, output):
    """Visualize predefined masks for GoPro cameras."""

    # Default resolution
    width, height = 2704, 2028

    if video:
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        cap.release()
        if not ret:
            print(f"Failed to read frame {frame} from {video}")
            return
        height, width = img.shape[:2]
        print(f"Using video frame: {width}x{height}")
    else:
        img = None

    # Create side-by-side comparison
    if img is not None:
        img_gopro9 = img.copy()
        img_hero13 = img.copy()
    else:
        img_gopro9 = np.zeros((height, width, 3), dtype=np.uint8) + 50
        img_hero13 = np.zeros((height, width, 3), dtype=np.uint8) + 50

    img_gopro9 = visualize_gopro9_masks(img_gopro9)
    img_hero13 = visualize_hero13_masks(img_hero13)

    # Resize for display if too large
    max_display_width = 1200
    if width > max_display_width:
        scale = max_display_width / width
        img_gopro9 = cv2.resize(img_gopro9, None, fx=scale, fy=scale)
        img_hero13 = cv2.resize(img_hero13, None, fx=scale, fy=scale)

    # Combine side by side
    combined = np.hstack([img_gopro9, img_hero13])

    if output:
        cv2.imwrite(output, combined)
        print(f"Saved to {output}")

    # Display
    cv2.imshow("Mask Comparison (GoPro 9 | Hero 13)", combined)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
