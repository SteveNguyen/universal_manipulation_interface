#!/usr/bin/env python3
"""
Generate a ChArUco calibration board for camera calibration.
Outputs a PDF ready for printing.

Usage: python generate_charuco_board.py --output charuco_board.pdf
"""

import cv2
import numpy as np
import click
from pathlib import Path


def mm_to_pixels(mm, dpi=300):
    """Convert millimeters to pixels at given DPI."""
    inches = mm / 25.4
    return int(inches * dpi)


def create_charuco_board_image(rows, cols, square_size_mm, marker_size_mm,
                               dictionary, dpi=300, margin_mm=5):
    """
    Create a ChArUco board image.

    Args:
        rows: Number of squares (rows)
        cols: Number of squares (columns)
        square_size_mm: Size of each square in mm
        marker_size_mm: Size of ArUco marker in mm
        dictionary: ArUco dictionary name
        dpi: Dots per inch for printing
        margin_mm: Margin around board in mm

    Returns:
        numpy array: Board image
    """

    # Get ArUco dictionary
    aruco_dict_map = {
        '4x4_50': cv2.aruco.DICT_4X4_50,
        '4x4_100': cv2.aruco.DICT_4X4_100,
        '4x4_250': cv2.aruco.DICT_4X4_250,
        '4x4_1000': cv2.aruco.DICT_4X4_1000,
        '5x5_50': cv2.aruco.DICT_5X5_50,
        '5x5_100': cv2.aruco.DICT_5X5_100,
        '5x5_250': cv2.aruco.DICT_5X5_250,
        '5x5_1000': cv2.aruco.DICT_5X5_1000,
        '6x6_50': cv2.aruco.DICT_6X6_50,
        '6x6_100': cv2.aruco.DICT_6X6_100,
        '6x6_250': cv2.aruco.DICT_6X6_250,
        '6x6_1000': cv2.aruco.DICT_6X6_1000,
    }

    if dictionary not in aruco_dict_map:
        raise ValueError(f"Unknown dictionary: {dictionary}. Choose from: {list(aruco_dict_map.keys())}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_map[dictionary])

    # Create CharucoBoard
    board = cv2.aruco.CharucoBoard(
        (cols, rows),
        square_size_mm / 1000.0,  # Convert to meters
        marker_size_mm / 1000.0,   # Convert to meters
        aruco_dict
    )

    # Calculate image size in pixels
    square_size_px = mm_to_pixels(square_size_mm, dpi)
    margin_px = mm_to_pixels(margin_mm, dpi)

    # Board dimensions
    board_width_px = cols * square_size_px
    board_height_px = rows * square_size_px

    # Total image dimensions with margins
    img_width = board_width_px + 2 * margin_px
    img_height = board_height_px + 2 * margin_px

    # Generate board image
    board_img = board.generateImage((board_width_px, board_height_px), marginSize=0, borderBits=1)

    # Create white background with margins
    full_img = np.ones((img_height, img_width), dtype=np.uint8) * 255

    # Place board in center
    y_offset = margin_px
    x_offset = margin_px
    full_img[y_offset:y_offset+board_height_px, x_offset:x_offset+board_width_px] = board_img

    return full_img


def add_info_text(img, rows, cols, square_size_mm, marker_size_mm, dictionary, dpi=300):
    """Add information text to the image."""

    # Convert to color for colored text
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Info text
    info_lines = [
        f"ChArUco Calibration Board",
        f"",
        f"Board size: {cols} x {rows} squares",
        f"Square size: {square_size_mm} mm",
        f"Marker size: {marker_size_mm} mm",
        f"Dictionary: {dictionary}",
        f"",
    ]

    # Calculate text position (bottom of image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 1
    line_height = 25

    y_start = img_color.shape[0] - len(info_lines) * line_height - 15
    x_start = 65

    for i, line in enumerate(info_lines):
        y = y_start + i * line_height
        # if "IMPORTANT" in line or line.startswith("Board size"):
        color = (0, 0, 255)  # Red for important
        thickness_line = 2
        # else:
            # color = (0, 0, 0)  # Black
            # thickness_line = 1

        cv2.putText(img_color, line, (x_start, y), font, font_scale,
                   color, thickness_line, cv2.LINE_AA)

    return img_color


@click.command()
@click.option('--rows', type=int, default=7,
              help='Number of squares (rows)')
@click.option('--cols', type=int, default=5,
              help='Number of squares (columns)')
@click.option('--square_size', type=float, default=40.0,
              help='Size of each square in millimeters')
@click.option('--marker_size', type=float, default=25.0,
              help='Size of ArUco marker in millimeters ')
@click.option('--dictionary',
              type=click.Choice(['4x4_50', '4x4_100', '4x4_250', '4x4_1000',
                               '5x5_50', '5x5_100', '5x5_250', '5x5_1000',
                               '6x6_50', '6x6_100', '6x6_250', '6x6_1000']),
              default='4x4_50',
              help='ArUco dictionary to use')
@click.option('--dpi', type=int, default=300,
              help='Resolution for printing (300 is standard)')
@click.option('--output', default='charuco_board.pdf',
              help='Output PDF filename')
@click.option('--no_info', is_flag=True,
              help='Do not add information text to board')
def main(rows, cols, square_size, marker_size, dictionary, dpi, output, no_info):
    """Generate a ChArUco calibration board for camera calibration."""

    print("=" * 60)
    print("ChArUco Board Generator")
    print("=" * 60)
    print()
    print(f"Board Configuration:")
    print(f"  Rows: {rows}")
    print(f"  Columns: {cols}")
    print(f"  Square size: {square_size} mm")
    print(f"  Marker size: {marker_size} mm")
    print(f"  Dictionary: {dictionary}")
    print(f"  DPI: {dpi}")
    print()

    # Validate marker size
    if marker_size >= square_size:
        print("⚠ WARNING: Marker size should be smaller than square size!")
        print(f"  Recommended: {square_size * 0.8:.1f} mm (80% of square size)")
        print()

    # Generate board image
    print("Generating board image...")
    try:
        board_img = create_charuco_board_image(
            rows, cols, square_size, marker_size, dictionary, dpi
        )
        print(f"✓ Board image generated: {board_img.shape[1]}x{board_img.shape[0]} pixels")
    except Exception as e:
        print(f"✗ Error generating board: {e}")
        return 1

    # Add info text if requested
    if not no_info:
        print("Adding information text...")
        board_img = add_info_text(
            board_img, rows, cols, square_size, marker_size, dictionary, dpi
        )

    # Save as PNG first (for preview)
    output_path = Path(output)
    png_path = output_path.with_suffix('.png')

    print(f"Saving PNG preview: {png_path}")
    cv2.imwrite(str(png_path), board_img)

    # Convert to PDF
    try:
        from PIL import Image

        print(f"Converting to PDF: {output}")

        # Convert numpy array to PIL Image
        if len(board_img.shape) == 3:
            pil_img = Image.fromarray(cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(board_img)

        # Calculate page size based on DPI
        # A4 is 210 x 297 mm
        page_width_mm = 210
        page_height_mm = 297

        # Calculate image size in mm at given DPI
        img_width_mm = (board_img.shape[1] / dpi) * 25.4
        img_height_mm = (board_img.shape[0] / dpi) * 25.4

        # Check if it fits on A4
        if img_width_mm > page_width_mm or img_height_mm > page_height_mm:
            print("⚠ WARNING: Board may not fit on A4 paper!")
            print(f"  Board size: {img_width_mm:.1f} x {img_height_mm:.1f} mm")
            print(f"  A4 size: {page_width_mm} x {page_height_mm} mm")
            print()

        # Save as PDF
        pil_img.save(str(output), "PDF", resolution=dpi)

        print()
        print("=" * 60)
        print("✓ ChArUco board generated successfully!")
        print("=" * 60)
        print()
        print(f"Files created:")
        print(f"  PDF (for printing): {output}")
        print(f"  PNG (for preview):  {png_path}")
        print()
        print("Printing instructions:")
        print("1. Print the PDF at 100% scale (no scaling/fit-to-page)")
        print("2. Use high-quality printer (inkjet or laser)")
        print("3. Print on regular paper (not photo paper)")
        print("4. MEASURE the actual square size after printing")
        print("5. Mount on rigid flat surface (foam board, cardboard)")
        print()
        print("Calibration command:")
        print(f"  python calibrate_hero13.py \\")
        print(f"    --video calib_video.mp4 \\")
        print(f"    --pattern charuco \\")
        print(f"    --rows {rows} \\")
        print(f"    --cols {cols} \\")
        print(f"    --square_size {square_size/1000:.3f} \\")
        print(f"    --marker_size {marker_size/1000:.3f}")
        print()
        print("⚠ IMPORTANT: Measure actual printed square size!")
        print("   Use measured value (in meters) for --square_size")
        print()

        return 0

    except ImportError:
        print()
        print("✗ PIL (Pillow) not installed. Cannot generate PDF.")
        print("  Install with: pip install Pillow")
        print()
        print(f"PNG file saved: {png_path}")
        print("You can convert PNG to PDF using other tools.")
        return 1
    except Exception as e:
        print(f"✗ Error creating PDF: {e}")
        return 1


if __name__ == '__main__':
    main()
