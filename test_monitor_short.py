#!/usr/bin/env python3
"""Quick end-to-end test of the full parking monitor pipeline on a short clip."""

import cv2
import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from car_detector import CarDetector
from vehicle_tracker import VehicleTracker
from search_analyzer import SearchAnalyzer
from spot_occupancy import SpotOccupancyChecker

VIDEO_PATH = "/home/joe/Parking_App/data/Footage/Parking_30min_data_gathering.mp4"
SPOTS_PATH = "/home/joe/Parking_App/parking_vision_project/data/configs/parking_spots.json"
OUTPUT_DIR = "/home/joe/Parking_App/parking_vision_project/output"

# Process frames at these video positions (frame numbers)
TEST_FRAMES = [300, 600, 900, 1500, 3000]


def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def main():
    import numpy as np

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    with open(SPOTS_PATH) as f:
        spots_config = json.load(f)
    spots = spots_config['spots']
    print(f"Loaded {len(spots)} spots")

    # Initialize all components
    detector = CarDetector()
    spot_checker = SpotOccupancyChecker()
    tracker = VehicleTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    search_analyzer = SearchAnalyzer(
        search_timeout=30.0, min_frames_threshold=15, alert_threshold=20.0
    )

    # Compute ROI
    roi = SpotOccupancyChecker.compute_roi(spots, (height, width), padding=150)
    print(f"Detection ROI: {roi}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nProcessing {len(TEST_FRAMES)} frames: {TEST_FRAMES}")
    print("=" * 70)

    for frame_idx, frame_num in enumerate(TEST_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"  Frame {frame_num}: FAILED to read")
            continue

        video_time = frame_num / fps
        t0 = time.time()

        # 1. YOLO detection (ROI-cropped)
        detections = detector.detect_cars(frame, roi=roi)
        tracker_dets = detector.detect_cars_for_tracker(frame, roi=roi)
        t_yolo = time.time() - t0

        # 2. Vehicle tracking
        tracked = tracker.update(tracker_dets)

        # 3. Search analysis
        search_status = search_analyzer.update(tracked, spots, video_time)

        # 4. Spot occupancy (hybrid)
        yolo_occupied = set()
        for spot in spots:
            polygon = [(p[0], p[1]) for p in spot['points']]
            for det in detections:
                cx = det['bbox'][0] + det['bbox'][2] // 2
                cy = det['bbox'][1] + det['bbox'][3] // 2
                if point_in_polygon((cx, cy), polygon):
                    yolo_occupied.add(spot['id'])

        t1 = time.time()
        spot_results = spot_checker.check_spots(frame, spots)
        t_analysis = time.time() - t1

        analysis_occupied = {r['spot_id'] for r in spot_results if r['occupied']}
        combined = yolo_occupied | analysis_occupied
        empty = set(s['id'] for s in spots) - combined

        t_total = time.time() - t0

        print(f"\n  Frame {frame_num} (t={video_time:.1f}s):")
        print(f"    YOLO detections: {len(detections)} ({t_yolo:.1f}s)")
        print(f"    Spot analysis:   {len(analysis_occupied)}/46 occupied ({t_analysis:.1f}s)")
        print(f"    Combined:        {len(combined)}/46 occupied")
        print(f"    Empty spots:     {empty if empty else 'none'}")
        print(f"    Tracked vehicles: {len(tracked)}")
        print(f"    Searching:       {search_status.get('vehicles_currently_searching', 0)}")
        print(f"    Total time:      {t_total:.1f}s")

        # Draw annotated frame
        annotated = frame.copy()
        for spot in spots:
            pts = np.array(spot['points'], np.int32)
            sid = spot['id']
            color = (0, 0, 255) if sid in combined else (0, 255, 0)
            cv2.polylines(annotated, [pts], True, color, 3)
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(annotated, 0.75, overlay, 0.25, 0, annotated)
            cx = int(np.mean([p[0] for p in spot['points']]))
            cy = int(np.mean([p[1] for p in spot['points']]))
            cv2.putText(annotated, str(sid), (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for det in detections:
            x, y, w, h = det['bbox']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 100, 0), 2)
            cv2.putText(annotated, f"{det['confidence']:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

        # Draw ROI box
        rx1, ry1, rx2, ry2 = roi
        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

        cv2.putText(annotated,
                    f"Frame {frame_num} | Occupied: {len(combined)}/46 | "
                    f"YOLO: {len(detections)} | Empty: {empty if empty else 'none'}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out_path = os.path.join(OUTPUT_DIR, f"monitor_test_f{frame_num}.jpg")
        cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

    cap.release()

    # Final stats
    tracker_stats = tracker.get_statistics()
    search_stats = search_analyzer.get_statistics()
    print("\n" + "=" * 70)
    print("FINAL STATISTICS:")
    print(f"  Total vehicles detected: {tracker_stats.get('total_vehicles_detected', 0)}")
    print(f"  Active tracks: {tracker_stats.get('active_tracks', 0)}")
    print(f"  Parked: {search_stats.get('successful_parks', 0)}")
    print(f"  Failed searches: {search_stats.get('failed_searches', 0)}")
    print(f"\nOutput frames saved to: {OUTPUT_DIR}/monitor_test_f*.jpg")


if __name__ == "__main__":
    main()
