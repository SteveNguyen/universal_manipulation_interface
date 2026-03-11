#!/usr/bin/env python3
"""
Grabette Mask Tuner - Adjust mask coordinates and preview results.

The SLAM mask tells ORB-SLAM3 to ignore pixels occupied by the device body
so it doesn't track features that move with the camera.

Usage:
    .venv/bin/python scripts/grabette_mask_tuner.py --video rpi_data/grabette1/raw_video.mp4
    .venv/bin/python scripts/grabette_mask_tuner.py --video rpi_data/grabette1/raw_video.mp4 --frame 300

Edit the DEVICE_BODY_PTS coordinates below, then run the script to preview.
"""

import cv2
import numpy as np
import click
from pathlib import Path

# ============================================================
# EDIT THESE COORDINATES TO ADJUST THE MASK
# All coordinates are in pixels at 1296x972 (native RPi resolution)
# ============================================================

# Reference resolution (don't change)
REF_WIDTH = 1296
REF_HEIGHT = 972

# DEVICE BODY mask - covers the grabette housing visible in the frame.
# The white cylindrical body and arm are visible at bottom-right.
DEVICE_BODY_PTS = [
    [120, 972],     # bottom, start of device edge
    [280, 750],     # right side of curved arm
    [1030, 610],    # top of device body
    [1160, 780],    # top-right corner area
    [1296, 780],    # top-right corner area
    [1296, 972],    # bottom-right corner
]

# ============================================================
# CODE BELOW - NO NEED TO EDIT
# ============================================================

@click.command()
@click.option('--video', '-v', type=click.Path(exists=True), required=True,
              help='Path to a Grabette video file')
@click.option('--frame', '-f', type=int, default=100,
              help='Frame number to extract')
@click.option('--output', '-o', type=click.Path(), default=None,
              help='Output path for preview image')
def main(video, frame, output):
    """Preview the Grabette SLAM mask on a video frame."""

    # Load video frame
    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame} from {video}")
        print(f"Video has {total_frames} frames")
        return

    h, w = img.shape[:2]
    print(f"Video: {video}")
    print(f"Frame: {frame}/{total_frames}")
    print(f"Resolution: {w}x{h}")

    # Scale factor if resolution differs from reference
    scale_x = w / REF_WIDTH
    scale_y = h / REF_HEIGHT

    # Scale mask coordinates
    device_pts = np.array(DEVICE_BODY_PTS, dtype=np.float64)
    device_pts[:, 0] *= scale_x
    device_pts[:, 1] *= scale_y
    device_pts = device_pts.astype(np.int32)

    # Create visualizations
    original = img.copy()
    masked = img.copy()

    # Apply mask
    cv2.fillPoly(masked, [device_pts], (0, 0, 0))

    # Create overlay showing mask region
    overlay = original.copy()
    mask_highlight = np.zeros_like(original)
    cv2.fillPoly(mask_highlight, [device_pts], (0, 0, 255))  # Red fill
    overlay = cv2.addWeighted(overlay, 0.7, mask_highlight, 0.3, 0)
    cv2.polylines(overlay, [device_pts], True, (0, 255, 0), 2)  # Green outline

    # Add coordinate labels
    for i, pt in enumerate(device_pts):
        cv2.circle(overlay, tuple(pt), 6, (0, 255, 255), -1)
        label = f"P{i}: ({DEVICE_BODY_PTS[i][0]}, {DEVICE_BODY_PTS[i][1]})"
        # Offset label to the left if point is near right edge
        lx = pt[0] - 250 if pt[0] > w - 300 else pt[0] + 10
        ly = pt[1] - 10 if pt[1] > h - 30 else pt[1] + 5
        cv2.putText(overlay, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Add legend
    cv2.putText(overlay, "Grabette SLAM Mask Preview", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, "Red = masked region, Green = outline", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(overlay, "Edit DEVICE_BODY_PTS in script to adjust", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.putText(masked, "After masking", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Combine side by side
    combined = np.hstack([overlay, masked])

    # Resize for display if too large
    display_max_width = 2400
    if combined.shape[1] > display_max_width:
        scale = display_max_width / combined.shape[1]
        combined_display = cv2.resize(combined, None, fx=scale, fy=scale)
    else:
        combined_display = combined

    # Save output
    if output is None:
        output = '/tmp/grabette_mask_preview.png'
    cv2.imwrite(output, combined)
    print(f"\nSaved full resolution preview: {output}")

    # Also save a smaller version for quick viewing
    small_output = output.replace('.png', '_small.png')
    cv2.imwrite(small_output, combined_display)
    print(f"Saved display version: {small_output}")

    # Print coordinates for copy-paste
    print("\n" + "="*60)
    print("Current mask coordinates (edit DEVICE_BODY_PTS to change):")
    print("="*60)
    for i, pt in enumerate(DEVICE_BODY_PTS):
        print(f"  P{i}: ({pt[0]}, {pt[1]})")

    print("\n" + "="*60)
    print("To adjust the mask:")
    print("="*60)
    print("1. Edit DEVICE_BODY_PTS in this script")
    print("2. Re-run the script to see the new mask")
    print("3. Repeat until satisfied")
    print("\nGoal: Mask the device body/housing visible in frame")
    print("      so ORB-SLAM3 ignores features on the device")

    # Try to display
    try:
        cv2.imshow("Grabette Mask Preview (press any key to close)", combined_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print(f"\nCannot display window. View the saved image: {output}")


if __name__ == "__main__":
    main()
