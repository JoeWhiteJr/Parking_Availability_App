#!/usr/bin/env python3
"""
Extract parking spot positions from red dots drawn on a frame image.
Each red dot marks the center of a parking stall.
"""

import cv2
import numpy as np
import json
import os

DRAWN_PATH = "/home/joe/Parking_App/parking_vision_project/data/frame_grid_drawn_dots.jpg"
ORIGINAL_PATH = "/home/joe/Parking_App/parking_vision_project/output/frame_grid.jpg"
VIDEO_PATH = "/home/joe/Parking_App/data/Footage/Parking_30min_data_gathering.mp4"
OUTPUT_PATH = "/home/joe/Parking_App/parking_vision_project/data/configs/parking_spots.json"
OUTPUT_DIR = "/home/joe/Parking_App/parking_vision_project/output"


def find_homography(drawn_img, original_img):
    """Find homography from drawn image to original frame using ORB features."""
    gray_d = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2GRAY)
    gray_o = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(gray_d, None)
    kp2, des2 = orb.detectAndCompute(gray_o, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"Feature matching: {len(good)} good matches")

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    print(f"Homography inliers: {mask.ravel().sum()}")
    return H


def find_red_dots(img):
    """Detect red dots and return their centers."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red wraps around in HSV — need two ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2

    # Clean up
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50:  # too small, noise
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        dots.append((cx, cy, area))

    # Sort top-to-bottom, then left-to-right (by row buckets)
    dots.sort(key=lambda d: (d[1] // 40, d[0]))

    print(f"Found {len(dots)} red dots")
    return dots, red_mask


def get_spot_size(cy):
    """Return (half_width, half_height) based on y-position in original frame.
    Further from camera (lower y) = smaller spots due to perspective."""
    if cy < 2120:
        return (55, 45)
    elif cy < 2200:
        return (65, 55)
    elif cy < 2300:
        return (75, 65)
    elif cy < 2420:
        return (85, 75)
    elif cy < 2550:
        return (95, 85)
    else:
        return (110, 95)


def main():
    drawn = cv2.imread(DRAWN_PATH)
    original = cv2.imread(ORIGINAL_PATH)

    if drawn is None or original is None:
        print("ERROR: Cannot read images")
        return

    print(f"Drawn: {drawn.shape[1]}x{drawn.shape[0]}")
    print(f"Original: {original.shape[1]}x{original.shape[0]}")

    # Step 1: Homography
    H = find_homography(drawn, original)

    # Step 2: Find red dots in drawn image
    dots, red_mask = find_red_dots(drawn)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "red_mask.jpg"), red_mask)

    # Step 3: Transform dot centers to original frame coordinates
    spots = []
    for i, (dx, dy, area) in enumerate(dots):
        # Transform point
        pt = np.float32([[[dx, dy]]])
        pt_orig = cv2.perspectiveTransform(pt, H)
        ox, oy = int(pt_orig[0][0][0]), int(pt_orig[0][0][1])

        # Get spot size for this y-position
        hw, hh = get_spot_size(oy)

        spot = {
            "id": i + 1,
            "points": [
                [ox - hw, oy - hh],
                [ox + hw, oy - hh],
                [ox + hw, oy + hh],
                [ox - hw, oy + hh],
            ],
            "type": "normal",
        }
        spots.append(spot)
        print(f"  Dot {i+1}: drawn=({dx},{dy}) -> orig=({ox},{oy}) -> {hw*2}x{hh*2} spot")

    # Step 4: Save JSON
    config = {
        "video_path": VIDEO_PATH,
        "frame_size": [original.shape[1], original.shape[0]],
        "spots": spots,
        "total_spots": len(spots),
        "spot_types": {"normal": len(spots), "electric": 0, "reserved": 0},
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved {len(spots)} spots to {OUTPUT_PATH}")

    # Step 5: Verification image
    verify = original.copy()
    for spot in spots:
        pts = np.array(spot["points"], np.int32)
        cv2.polylines(verify, [pts], True, (0, 255, 0), 3)
        overlay = verify.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(verify, 0.8, overlay, 0.2, 0, verify)
        cx = int(np.mean([p[0] for p in spot["points"]]))
        cy = int(np.mean([p[1] for p in spot["points"]]))
        cv2.putText(verify, str(spot["id"]), (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out_path = os.path.join(OUTPUT_DIR, "spots_from_dots.jpg")
    cv2.imwrite(out_path, verify, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Saved verification: {out_path}")

    # Step 6: Debug — show detected dots on drawn image
    debug = drawn.copy()
    for i, (dx, dy, area) in enumerate(dots):
        cv2.circle(debug, (dx, dy), 8, (255, 0, 0), 2)
        cv2.putText(debug, str(i + 1), (dx + 10, dy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    debug_path = os.path.join(OUTPUT_DIR, "dots_detected.jpg")
    cv2.imwrite(debug_path, debug, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Saved dot detection debug: {debug_path}")


if __name__ == "__main__":
    main()
