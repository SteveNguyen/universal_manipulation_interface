#!/usr/bin/env python3
"""Check SLAM tracking percentage from trajectory CSV."""

import sys
import pandas as pd

def check_tracking(trajectory_csv):
    """Calculate and display tracking percentage."""
    try:
        df = pd.read_csv(trajectory_csv)
    except FileNotFoundError:
        print(f"❌ Error: {trajectory_csv} not found!")
        print("SLAM may have failed or not completed yet.")
        sys.exit(1)

    if 'is_lost' not in df.columns:
        print(f"❌ Error: 'is_lost' column not found in {trajectory_csv}")
        sys.exit(1)

    tracked = (~df['is_lost']).sum()
    total = len(df)
    pct = (tracked / total) * 100

    print("=" * 60)
    print("SLAM Tracking Results")
    print("=" * 60)
    print()
    print(f"Tracked frames: {tracked}/{total}")
    print(f"Tracking percentage: {pct:.1f}%")
    print()
    print("Comparison:")
    print(f"  Baseline (GoPro 9 T_b_c):  91.4%")
    print(f"  This run (Hero 13 T_b_c):  {pct:.1f}%")
    print()

    if pct >= 91.4:
        print("✅ SUCCESS! Hero 13 calibration performs as well or better!")
    elif pct >= 80:
        print("✅ GOOD! Above 80% threshold, acceptable performance.")
    elif pct >= 50:
        print("⚠️  MODERATE: Above 50% but could be better.")
    else:
        print("❌ POOR: Below 50% tracking, calibration may need improvement.")
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        trajectory_csv = sys.argv[1]
    else:
        trajectory_csv = "test_slam_hero13_proper/output/trajectory.csv"

    check_tracking(trajectory_csv)
