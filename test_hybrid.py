#!/usr/bin/env python3
"""Test hybrid detection: SAHI tiled YOLO + spot-based occupancy analysis."""

import cv2
import numpy as np
import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from car_detector import CarDetector
from spot_occupancy import SpotOccupancyChecker

VIDEO_PATH = "/home/joe/Parking_App/data/Footage/Parking_30min_data_gathering.mp4"
SPOTS_PATH = "/home/joe/Parking_App/parking_vision_project/data/configs/parking_spots.json"
OUTPUT_DIR = "/home/joe/Parking_App/parking_vision_project/output"


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return

    with open(SPOTS_PATH) as f:
        spots_config = json.load(f)
    spots = spots_config['spots']
    print(f"Loaded {len(spots)} spots")

    detector = CarDetector()
    spot_checker = SpotOccupancyChecker()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Cannot read frame")
        return

    # Compute ROI from spot bounding box (parking lot area only)
    roi = SpotOccupancyChecker.compute_roi(spots, frame.shape[:2], padding=150)
    print(f"Detection ROI: {roi}")

    # --- SAHI tiled YOLO detection (restricted to parking lot ROI) ---
    print("\n--- SAHI Tiled YOLO Detection (ROI-cropped) ---")
    t0 = time.time()
    detections = detector.detect_cars(frame, roi=roi)
    t1 = time.time()
    print(f"Detections: {len(detections)} (took {t1-t0:.1f}s)")

    # Count which spots have YOLO hits
    yolo_occupied = set()
    for spot in spots:
        polygon = [(p[0], p[1]) for p in spot['points']]
        for det in detections:
            cx = det['bbox'][0] + det['bbox'][2] // 2
            cy = det['bbox'][1] + det['bbox'][3] // 2
            n = len(polygon)
            inside = False
            p1x, p1y = polygon[0]
            for i in range(1, n + 1):
                p2x, p2y = polygon[i % n]
                if cy > min(p1y, p2y):
                    if cy <= max(p1y, p2y):
                        if cx <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (cy - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or cx <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            if inside:
                yolo_occupied.add(spot['id'])

    print(f"YOLO spots occupied: {len(yolo_occupied)}/{len(spots)}")

    # --- Spot-based image analysis ---
    print("\n--- Spot-Based Image Analysis ---")
    t0 = time.time()
    spot_results = spot_checker.check_spots(frame, spots)
    t1 = time.time()
    print(f"Analysis took {t1-t0:.1f}s")

    analysis_occupied = set()
    for r in spot_results:
        if r['occupied']:
            analysis_occupied.add(r['spot_id'])
        status = "OCCUPIED" if r['occupied'] else "empty"
        print(f"  Spot {r['spot_id']:>2}: {status:>8}  edge={r['edge_score']:>5.1f}  "
              f"bright={r['mean_brightness']:>5.1f}  var={r['variance_score']:>7.1f}  "
              f"std={r['gray_std']:>5.1f}  conf={r['confidence']:.2f}")

    print(f"\nSpot analysis occupied: {len(analysis_occupied)}/{len(spots)}")

    # --- Combined ---
    combined = yolo_occupied | analysis_occupied
    print(f"\nCombined occupied: {len(combined)}/{len(spots)}")
    print(f"  YOLO only: {yolo_occupied - analysis_occupied}")
    print(f"  Analysis only: {analysis_occupied - yolo_occupied}")
    print(f"  Both agree: {yolo_occupied & analysis_occupied}")
    empty = set(s['id'] for s in spots) - combined
    print(f"  Empty spots: {empty}")

    # --- Draw verification ---
    annotated = frame.copy()
    for spot in spots:
        pts = np.array(spot['points'], np.int32)
        sid = spot['id']
        if sid in combined:
            color = (0, 0, 255)  # Red = occupied
        else:
            color = (0, 255, 0)  # Green = empty

        cv2.polylines(annotated, [pts], True, color, 3)
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(annotated, 0.75, overlay, 0.25, 0, annotated)

        cx = int(np.mean([p[0] for p in spot['points']]))
        cy = int(np.mean([p[1] for p in spot['points']]))
        cv2.putText(annotated, str(sid), (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw YOLO detections as blue boxes
    for det in detections:
        x, y, w, h = det['bbox']
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 100, 0), 2)

    cv2.putText(annotated, f"Occupied: {len(combined)}/{len(spots)} | "
                f"YOLO: {len(yolo_occupied)} | Analysis: {len(analysis_occupied)}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    out_path = os.path.join(OUTPUT_DIR, "hybrid_test.jpg")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
