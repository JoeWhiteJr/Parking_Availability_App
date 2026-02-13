#!/usr/bin/env python3
"""
Spot-based occupancy detection for parking lots.

Instead of detecting cars as objects, this analyzes each defined parking spot
region directly — comparing edge density, texture, color variance, and mean
brightness against empty pavement thresholds to determine if a car is present.

Works well with fixed cameras and pre-defined spot polygons.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class SpotOccupancyChecker:
    """
    Determines parking spot occupancy by analyzing the pixel content within
    each spot's polygon region.

    Key signals:
    - Edge density: cars have sharp edges from body panels, windows, wheels
    - Mean brightness: empty pavement is bright (~150-200 gray), cars are darker (~60-140)
    - Color variance: cars have varied colors, but painted markings can too
    - Grayscale std: texture measure
    """

    def __init__(self, edge_threshold: float = 15.0):
        """
        Args:
            edge_threshold: Mean Canny edge value above which a spot likely has a car.
                            Typical values: 12-20 for angled parking lot cameras.
        """
        self.edge_threshold = edge_threshold
        self.spot_baselines: Dict[int, Dict] = {}

    @staticmethod
    def compute_roi(spots: List[Dict], frame_shape: Tuple[int, int],
                    padding: int = 100) -> Tuple[int, int, int, int]:
        """
        Compute a bounding-box ROI that encloses all parking spots.

        Args:
            spots: List of spot dicts with 'points'
            frame_shape: (height, width) of the frame
            padding: Extra pixels around the bounding box

        Returns:
            (x1, y1, x2, y2) ROI rectangle in frame coordinates
        """
        all_x = []
        all_y = []
        for spot in spots:
            for p in spot['points']:
                all_x.append(p[0])
                all_y.append(p[1])

        h, w = frame_shape[:2]
        x1 = max(0, min(all_x) - padding)
        y1 = max(0, min(all_y) - padding)
        x2 = min(w, max(all_x) + padding)
        y2 = min(h, max(all_y) + padding)
        return (x1, y1, x2, y2)

    def check_spots(self, frame: np.ndarray, spots: List[Dict]) -> List[Dict]:
        """
        Check occupancy of all spots using image analysis.

        Args:
            frame: Current video frame (BGR)
            spots: List of spot dicts with 'id' and 'points'

        Returns:
            List of dicts with per-spot analysis results
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        results = []
        for spot in spots:
            spot_id = spot['id']
            points = np.array(spot['points'], np.int32)

            # Create mask for this spot's polygon
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)

            # Count pixels inside spot
            pixel_count = cv2.countNonZero(mask)
            if pixel_count == 0:
                results.append({
                    'spot_id': spot_id, 'occupied': False,
                    'confidence': 0.0, 'edge_score': 0.0,
                    'variance_score': 0.0, 'gray_std': 0.0,
                    'mean_brightness': 255.0,
                })
                continue

            # --- Signal 1: Edge density ---
            spot_edges = cv2.bitwise_and(edges, edges, mask=mask)
            edge_score = cv2.countNonZero(spot_edges) / pixel_count * 255

            # --- Signal 2: Mean brightness ---
            gray_vals = gray[mask > 0]
            mean_brightness = float(np.mean(gray_vals))
            gray_std = float(np.std(gray_vals))

            # --- Signal 3: Color variance ---
            b, g, r = cv2.split(frame)
            b_vals = b[mask > 0]
            g_vals = g[mask > 0]
            r_vals = r[mask > 0]
            variance_score = float(np.var(b_vals) + np.var(g_vals) + np.var(r_vals))

            # --- Occupancy decision ---
            # Two strong signals for a parked car:
            # 1. Dark spot (car body absorbs/blocks light → low brightness)
            # 2. Edge detail (car body, windows, wheels → high edge density)
            #
            # Empty pavement (even with painted markings like handicapped symbols)
            # has low edges and moderate-to-high brightness.
            if mean_brightness < 55.0:
                # Very dark region — car is present regardless of edge count
                occupied = True
            elif edge_score > self.edge_threshold:
                # Significant edge detail — car body, reflections, etc.
                occupied = True
            else:
                # Not dark + not enough edges = empty pavement
                occupied = False

            # Confidence score
            if occupied:
                edge_factor = min(1.0, edge_score / self.edge_threshold)
                dark_factor = min(1.0, max(0, (55.0 - mean_brightness)) / 25.0)
                confidence = min(1.0, max(0.3, edge_factor * 0.5 + dark_factor * 0.3 + (gray_std / 50.0) * 0.2))
            else:
                bright_factor = min(1.0, max(0, (mean_brightness - 55.0)) / 40.0)
                low_edge = min(1.0, max(0, (self.edge_threshold - edge_score)) / self.edge_threshold)
                confidence = max(0.2, min(1.0, bright_factor * 0.5 + low_edge * 0.5))

            results.append({
                'spot_id': spot_id,
                'occupied': occupied,
                'confidence': round(confidence, 3),
                'edge_score': round(edge_score, 1),
                'variance_score': round(variance_score, 1),
                'gray_std': round(gray_std, 1),
                'mean_brightness': round(mean_brightness, 1),
            })

        return results

    def calibrate_empty(self, frame: np.ndarray, spots: List[Dict]):
        """
        Calibrate baselines using a frame where spots are known to be empty.
        Call this once with an empty lot frame to improve accuracy.
        """
        results = self.check_spots(frame, spots)
        for r in results:
            self.spot_baselines[r['spot_id']] = {
                'edge_baseline': r['edge_score'],
                'variance_baseline': r['variance_score'],
                'brightness_baseline': r['mean_brightness'],
            }
