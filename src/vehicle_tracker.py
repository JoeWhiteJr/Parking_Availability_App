#!/usr/bin/env python3
"""
SORT-based Vehicle Tracker for Parking Lot Monitoring

Uses Kalman Filter for motion prediction and Hungarian Algorithm
for detection-to-track association to maintain vehicle identity across frames.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import time


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.

    Args:
        bbox1: First bbox [x1, y1, x2, y2]
        bbox2: Second bbox [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = bbox1_area + bbox2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    Convert [x1, y1, x2, y2] to [x_center, y_center, area, aspect_ratio]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    area = w * h
    aspect_ratio = w / float(h) if h > 0 else 1.0
    return np.array([x, y, area, aspect_ratio]).reshape((4, 1))


def convert_z_to_bbox(z: np.ndarray) -> np.ndarray:
    """
    Convert [x_center, y_center, area, aspect_ratio] to [x1, y1, x2, y2]
    """
    w = np.sqrt(z[2] * z[3])
    h = z[2] / w if w > 0 else 0
    return np.array([
        z[0] - w / 2.0,
        z[1] - h / 2.0,
        z[0] + w / 2.0,
        z[1] + h / 2.0
    ]).flatten()


@dataclass
class TrackedVehicle:
    """Represents a tracked vehicle with its state and history"""
    track_id: int
    bbox: List[int]  # Current [x, y, w, h]
    centroid: Tuple[int, int]
    confidence: float
    first_seen_time: float
    last_seen_time: float
    frames_visible: int
    is_parked: bool = False
    parked_spot_id: Optional[int] = None
    is_searching: bool = False
    search_duration: float = 0.0
    centroid_history: List[Tuple[int, int]] = field(default_factory=list)


class KalmanBoxTracker:
    """
    Single object tracker using Kalman Filter.

    State vector: [x_center, y_center, area, aspect_ratio, vx, vy, va]
    Measurement: [x_center, y_center, area, aspect_ratio]
    """
    count = 0

    def __init__(self, bbox: np.ndarray):
        """
        Initialize tracker with bounding box [x1, y1, x2, y2]
        """
        # Kalman filter with 7 state variables, 4 measurements
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0

        # Covariance matrix
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state with first detection
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.confidence = 0.0

    def update(self, bbox: np.ndarray, confidence: float = 0.0):
        """Update state with new detection"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.last_seen = time.time()
        self.confidence = confidence
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        """Advance state and return predicted bbox"""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(convert_z_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        """Return current state as [x1, y1, x2, y2]"""
        return convert_z_to_bbox(self.kf.x)


class VehicleTracker:
    """
    SORT-based multi-object tracker for vehicles.

    Maintains vehicle identity across frames using IoU-based association
    and Kalman filter prediction.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Args:
            max_age: Maximum frames to keep track alive without detections
            min_hits: Minimum hits before track is considered confirmed
            iou_threshold: Minimum IoU for detection-to-track association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.total_vehicles = 0

    def update(self, detections: np.ndarray) -> List[TrackedVehicle]:
        """
        Update tracker with new detections.

        Args:
            detections: numpy array of shape (N, 5) where each row is
                       [x1, y1, x2, y2, confidence]

        Returns:
            List of TrackedVehicle objects for confirmed tracks
        """
        self.frame_count += 1

        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []

        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)

        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, trks
        )

        # Update matched trackers with detections
        for m in matched:
            self.trackers[m[1]].update(detections[m[0], :4], detections[m[0], 4])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i, :4])
            trk.confidence = detections[i, 4]
            self.trackers.append(trk)
            self.total_vehicles += 1

        # Build output list of tracked vehicles
        tracked_vehicles = []
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            d = trk.get_state()

            # Only return confirmed tracks (enough hits and recently updated)
            if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = [int(d[0]), int(d[1]), int(d[2] - d[0]), int(d[3] - d[1])]
                centroid = (int((d[0] + d[2]) / 2), int((d[1] + d[3]) / 2))

                vehicle = TrackedVehicle(
                    track_id=trk.id,
                    bbox=bbox,
                    centroid=centroid,
                    confidence=trk.confidence,
                    first_seen_time=trk.first_seen,
                    last_seen_time=trk.last_seen,
                    frames_visible=trk.hits
                )
                tracked_vehicles.append(vehicle)

            i -= 1

            # Remove dead tracks
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return tracked_vehicles

    def _associate_detections_to_trackers(
        self,
        detections: np.ndarray,
        trackers: np.ndarray
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Associate detections to tracked objects using Hungarian algorithm.

        Returns:
            matched: Array of matched (detection_idx, tracker_idx) pairs
            unmatched_detections: List of unmatched detection indices
            unmatched_trackers: List of unmatched tracker indices
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), []

        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), [], list(range(len(trackers)))

        # Build IoU cost matrix
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = iou(det[:4], trk[:4])

        # Hungarian algorithm (minimize negative IoU = maximize IoU)
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                row_indices, col_indices = linear_sum_assignment(-iou_matrix)
                matched_indices = np.array(list(zip(row_indices, col_indices)))
        else:
            matched_indices = np.empty(shape=(0, 2))

        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # Filter out low IoU matches
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, unmatched_detections, unmatched_trackers

    def get_statistics(self) -> Dict:
        """Get tracker statistics"""
        return {
            'total_vehicles_detected': self.total_vehicles,
            'active_tracks': len(self.trackers),
            'frame_count': self.frame_count
        }

    def reset(self):
        """Reset tracker state"""
        self.trackers = []
        self.frame_count = 0
        self.total_vehicles = 0
        KalmanBoxTracker.count = 0
