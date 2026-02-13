#!/usr/bin/env python3
"""Generate diagnostic frame showing detections + parking spot overlays."""

import cv2
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from car_detector import CarDetector

VIDEO_PATH = "/home/joe/Parking_App/data/Footage/Parking_30min_data_gathering.mp4"
SPOTS_PATH = "/home/joe/Parking_App/parking_vision_project/data/configs/parking_spots.json"
OUTPUT_DIR = "/home/joe/Parking_App/parking_vision_project/output"

def main():
    # Load video and grab a few sample frames
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, {fps:.1f} fps, {total_frames} frames ({total_frames/fps:.0f}s)")

    # Load parking spots
    spots = []
    if os.path.exists(SPOTS_PATH):
        with open(SPOTS_PATH, 'r') as f:
            spots_data = json.load(f)
            spots = spots_data.get('spots', [])
        print(f"Loaded {len(spots)} parking spot definitions")
    else:
        print("No parking spots config found")

    # Init detector (auto-loads YOLOv8 if ultralytics is installed)
    detector = CarDetector()
    if detector.yolo_model is not None:
        print("Using YOLOv8 for detection")
    else:
        print("WARNING: YOLOv8 not available, using basic fallback detection")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sample 3 frames: start (frame 100), middle, near end
    sample_points = [100, total_frames // 2, total_frames - 200]

    for i, frame_num in enumerate(sample_points):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {frame_num}")
            continue

        video_time = frame_num / fps
        print(f"\n--- Frame {frame_num} (video time: {video_time:.1f}s) ---")

        # Run detection
        detections = detector.detect_cars(frame)
        print(f"  Detections: {len(detections)}")
        for det in detections:
            print(f"    {det['class']}: conf={det['confidence']:.2f}, bbox={det['bbox']}")

        # Draw on frame
        annotated = frame.copy()

        # Draw parking spot polygons (green outlines with labels)
        for spot in spots:
            pts = np.array(spot['points'], np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 3)
            cx = int(np.mean([p[0] for p in spot['points']]))
            cy = int(np.mean([p[1] for p in spot['points']]))
            cv2.putText(annotated, f"Spot {spot['id']}", (cx - 40, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw detection boxes (red with confidence)
        for det in detections:
            x, y, w, h = det['bbox']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(annotated, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add header text
        cv2.putText(annotated, f"Frame {frame_num} | Time: {video_time:.1f}s | Detections: {len(detections)}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(annotated, "GREEN = defined spots | RED = detected vehicles",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        out_path = os.path.join(OUTPUT_DIR, f"diagnostic_frame_{i+1}.jpg")
        cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  Saved: {out_path}")

    cap.release()
    print("\nDone! Check output/diagnostic_frame_*.jpg")

if __name__ == "__main__":
    main()
