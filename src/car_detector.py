#!/usr/bin/env python3
"""
Car Detection System for Parking Lot Monitoring
Uses YOLOv8 object detection to identify cars in parking lot footage.
Falls back to basic contour analysis if ultralytics is unavailable.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import os


class CarDetector:
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.3):
        """
        Initialize car detector with YOLOv8 model

        Args:
            model_path: Path to YOLO model weights (optional, defaults to yolov8n.pt)
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path or "yolov8n.pt"
        self.yolo_model = None
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = 0.4

        # COCO vehicle class IDs
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.vehicle_class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        # Try to load YOLOv8
        self._load_yolov8()

    def _load_yolov8(self):
        """Attempt to load YOLOv8 model via ultralytics."""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.model_path)
            print(f"YOLOv8 model loaded: {self.model_path}")
        except ImportError:
            print("ultralytics not installed — falling back to basic contour detection")
            self.yolo_model = None
        except Exception as e:
            print(f"Failed to load YOLOv8 model: {e} — falling back to basic detection")
            self.yolo_model = None

    def detect_cars(self, frame: np.ndarray,
                    roi: tuple = None) -> List[Dict]:
        """
        Detect cars in the given frame using tiled (SAHI-style) inference.

        Slices the frame into overlapping tiles, runs YOLOv8 on each,
        then merges results with NMS. Falls back to basic detection if
        YOLOv8 is unavailable.

        Args:
            frame: Input image frame (BGR)
            roi: Optional (x1, y1, x2, y2) to restrict detection to a region.
                 Detections are returned in full-frame coordinates.

        Returns:
            List of detection dicts with keys: bbox [x, y, w, h], confidence, class
        """
        if self.yolo_model is not None:
            return self._detect_tiled(frame, roi=roi)
        return self._detect_basic(frame, roi=roi)

    def _detect_yolov8(self, frame: np.ndarray) -> List[Dict]:
        """Run YOLOv8 inference on a single image and return vehicle detections."""
        results = self.yolo_model(frame, conf=self.confidence_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in self.vehicle_classes:
                    continue
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                w = x2 - x1
                h = y2 - y1
                detections.append({
                    'bbox': [int(x1), int(y1), int(w), int(h)],
                    'confidence': conf,
                    'class': self.vehicle_class_names.get(cls_id, "vehicle"),
                })
        return detections

    def _detect_tiled(self, frame: np.ndarray,
                      tile_size: int = 640,
                      overlap: float = 0.25,
                      roi: tuple = None) -> List[Dict]:
        """
        SAHI-style tiled detection: slice the frame into overlapping tiles,
        run YOLOv8 on each tile, offset bboxes back to full-frame coords,
        then apply NMS to remove duplicates.

        Args:
            frame: Full input frame
            tile_size: Size of each square tile
            overlap: Fraction of overlap between adjacent tiles
            roi: Optional (x1, y1, x2, y2) to restrict tiling to a sub-region.
                 Returned bboxes are in full-frame coordinates.
        """
        h, w = frame.shape[:2]
        stride = int(tile_size * (1 - overlap))

        # Determine the region to tile over
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            rx1 = max(0, rx1)
            ry1 = max(0, ry1)
            rx2 = min(w, rx2)
            ry2 = min(h, ry2)
            region = frame[ry1:ry2, rx1:rx2]
        else:
            rx1, ry1 = 0, 0
            region = frame

        rh, rw = region.shape[:2]

        all_boxes = []   # [x1, y1, x2, y2] in full-frame coords
        all_confs = []
        all_classes = []

        # Run on the full ROI region (downscaled by YOLO internally) to catch large cars
        region_dets = self._detect_yolov8(region)
        for d in region_dets:
            bx, by, bw, bh = d['bbox']
            all_boxes.append([bx + rx1, by + ry1, bx + rx1 + bw, by + ry1 + bh])
            all_confs.append(d['confidence'])
            all_classes.append(d['class'])

        # Tile across the region
        for y0 in range(0, rh, stride):
            for x0 in range(0, rw, stride):
                x1 = min(x0, rw - tile_size) if x0 + tile_size > rw else x0
                y1 = min(y0, rh - tile_size) if y0 + tile_size > rh else y0
                x2 = min(x1 + tile_size, rw)
                y2 = min(y1 + tile_size, rh)

                if x2 - x1 < 100 or y2 - y1 < 100:
                    continue  # Skip tiny edge tiles

                tile = region[y1:y2, x1:x2]
                tile_dets = self._detect_yolov8(tile)

                for d in tile_dets:
                    bx, by, bw, bh = d['bbox']
                    # Offset to full-frame coordinates
                    fx = bx + x1 + rx1
                    fy = by + y1 + ry1
                    all_boxes.append([fx, fy, fx + bw, fy + bh])
                    all_confs.append(d['confidence'])
                    all_classes.append(d['class'])

        if not all_boxes:
            return []

        # Apply NMS to merge overlapping detections from different tiles
        boxes_arr = np.array(all_boxes, dtype=np.float32)
        confs_arr = np.array(all_confs, dtype=np.float32)

        indices = cv2.dnn.NMSBoxes(
            boxes_arr.tolist(),
            confs_arr.tolist(),
            self.confidence_threshold,
            self.nms_threshold,
        )

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, x2, y2 = all_boxes[i]
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    'confidence': float(all_confs[i]),
                    'class': all_classes[i],
                })

        return detections

    def _detect_basic(self, frame: np.ndarray, roi: tuple = None) -> List[Dict]:
        """
        Basic car detection using contour analysis.
        This is a last-resort fallback when YOLOv8 is unavailable.

        Area thresholds are sized for high-res video (~3840x2160 or similar).
        """
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            region = frame[ry1:ry2, rx1:rx2]
        else:
            rx1, ry1 = 0, 0
            region = frame

        detections = []

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 5000 < area < 200000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.8 < aspect_ratio < 3.5:
                    detections.append({
                        'bbox': [x + rx1, y + ry1, w, h],
                        'confidence': 0.4,
                        'class': 'car',
                    })
        return detections

    def detect_cars_for_tracker(self, frame: np.ndarray,
                               roi: tuple = None) -> np.ndarray:
        """
        Detect cars and return in format suitable for SORT tracker.

        Args:
            frame: Input image frame
            roi: Optional (x1, y1, x2, y2) to restrict detection region

        Returns:
            numpy array of shape (N, 5) where each row is [x1, y1, x2, y2, confidence]
        """
        detections = self.detect_cars(frame, roi=roi)

        if not detections:
            return np.empty((0, 5))

        tracker_detections = []
        for det in detections:
            x, y, w, h = det['bbox']
            confidence = det['confidence']
            tracker_detections.append([x, y, x + w, y + h, confidence])

        return np.array(tracker_detections)

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.

        Args:
            frame: Input image frame
            detections: List of detections from detect_cars()

        Returns:
            Frame with drawn bounding boxes
        """
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame


def main():
    """Test the car detection system"""
    print("Car Detection System Test")
    print("=" * 30)

    detector = CarDetector()

    if detector.yolo_model is not None:
        print("Using YOLOv8 detection")
    else:
        print("Using basic contour detection (fallback)")

    print("Ready to process video frames")
    print("\nTo use this detector:")
    print("1. Load a video frame")
    print("2. Call detector.detect_cars(frame)")
    print("3. Use detector.draw_detections() to visualize")


if __name__ == "__main__":
    main()
