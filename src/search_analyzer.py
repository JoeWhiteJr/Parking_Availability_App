#!/usr/bin/env python3
"""
Parking Search Analyzer

Detects vehicles that are searching for parking and identifies those
that fail to find parking within the configured timeout period.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import time
import logging


@dataclass
class ParkingSearchEvent:
    """Records a vehicle's parking search attempt"""
    track_id: int
    entry_time: float
    exit_time: Optional[float] = None
    search_duration: float = 0.0
    frames_visible: int = 0
    path: List[Tuple[int, int]] = field(default_factory=list)
    parked_spot_id: Optional[int] = None
    outcome: str = 'searching'  # 'searching', 'parked', 'exited_without_parking'


class SearchAnalyzer:
    """
    Analyzes vehicle tracking data to detect "didn't find parking" events.

    Uses time-based detection: if a vehicle is visible for longer than
    the timeout threshold without parking, it's marked as failed search.
    """

    def __init__(
        self,
        search_timeout: float = 30.0,
        min_frames_threshold: int = 15,
        alert_threshold: float = 20.0
    ):
        """
        Args:
            search_timeout: Seconds before marking as failed search
            min_frames_threshold: Minimum frames visible to count as search attempt
            alert_threshold: Seconds before generating warning alert
        """
        self.search_timeout = search_timeout
        self.min_frames_threshold = min_frames_threshold
        self.alert_threshold = alert_threshold

        self.active_searches: Dict[int, ParkingSearchEvent] = {}
        self.completed_searches: List[ParkingSearchEvent] = []
        self.vehicles_parked = 0
        self.vehicles_failed_search = 0

        self.logger = logging.getLogger(__name__)

    def point_in_polygon(self, point: Tuple[int, int], polygon: List[List[int]]) -> bool:
        """
        Check if point is inside polygon using ray casting algorithm.

        Args:
            point: (x, y) tuple
            polygon: List of [x, y] points defining polygon vertices

        Returns:
            True if point is inside polygon
        """
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

    def check_if_parked(
        self,
        centroid: Tuple[int, int],
        spots: List[Dict]
    ) -> Optional[int]:
        """
        Check if vehicle centroid is within any parking spot.

        Args:
            centroid: Vehicle center point (x, y)
            spots: List of parking spot dictionaries with 'points' and 'id'

        Returns:
            Spot ID if parked, None otherwise
        """
        for spot in spots:
            if self.point_in_polygon(centroid, spot['points']):
                return spot['id']
        return None

    def update(
        self,
        tracked_vehicles: List,
        spots: List[Dict],
        current_time: float
    ) -> Dict:
        """
        Update search analysis with current tracking state.

        Args:
            tracked_vehicles: List of TrackedVehicle objects
            spots: List of parking spot dictionaries
            current_time: Current video time in seconds

        Returns:
            Dict with current search status and alerts
        """
        alerts = []
        active_track_ids = set()

        for vehicle in tracked_vehicles:
            track_id = vehicle.track_id
            active_track_ids.add(track_id)
            centroid = vehicle.centroid

            # Check if vehicle is in a parking spot
            parked_spot_id = self.check_if_parked(centroid, spots)

            if track_id not in self.active_searches:
                # New vehicle - start tracking
                self.active_searches[track_id] = ParkingSearchEvent(
                    track_id=track_id,
                    entry_time=current_time
                )

            search_event = self.active_searches[track_id]
            search_event.frames_visible += 1
            search_event.path.append(centroid)
            search_event.search_duration = current_time - search_event.entry_time

            # Update vehicle state
            vehicle.search_duration = search_event.search_duration

            if parked_spot_id is not None:
                # Vehicle found parking
                if search_event.outcome == 'searching':
                    search_event.outcome = 'parked'
                    search_event.parked_spot_id = parked_spot_id
                    search_event.exit_time = current_time
                    self.vehicles_parked += 1
                    self.completed_searches.append(search_event)
                    del self.active_searches[track_id]
                    self.logger.info(
                        f"Vehicle {track_id} parked in spot {parked_spot_id} "
                        f"after {search_event.search_duration:.1f}s"
                    )

                vehicle.is_parked = True
                vehicle.parked_spot_id = parked_spot_id
                vehicle.is_searching = False
            else:
                # Vehicle is NOT in a parking spot
                vehicle.is_parked = False
                vehicle.parked_spot_id = None
                vehicle.is_searching = True

                # Check for alerts
                if search_event.search_duration > self.alert_threshold:
                    alert_level = 'warning'
                    if search_event.search_duration > self.search_timeout:
                        alert_level = 'critical'

                    alerts.append({
                        'track_id': track_id,
                        'search_duration': search_event.search_duration,
                        'alert_level': alert_level,
                        'centroid': centroid
                    })

        # Check for disappeared vehicles (left frame without parking)
        self._handle_disappeared_vehicles(active_track_ids, current_time)

        return {
            'active_searches': len(self.active_searches),
            'vehicles_currently_searching': sum(
                1 for v in tracked_vehicles if v.is_searching
            ),
            'vehicles_parked_total': self.vehicles_parked,
            'vehicles_failed_total': self.vehicles_failed_search,
            'alerts': alerts,
            'current_search_events': [
                {
                    'track_id': tid,
                    'searching_seconds': event.search_duration,
                    'entry_timestamp': datetime.fromtimestamp(event.entry_time).isoformat()
                }
                for tid, event in self.active_searches.items()
                if event.outcome == 'searching'
            ]
        }

    def _handle_disappeared_vehicles(
        self,
        active_track_ids: set,
        current_time: float
    ):
        """Handle vehicles that have left the frame"""
        disappeared_ids = []

        for track_id in self.active_searches:
            if track_id not in active_track_ids:
                disappeared_ids.append(track_id)

        for track_id in disappeared_ids:
            search_event = self.active_searches[track_id]
            search_event.exit_time = current_time

            # Only count if visible long enough
            if search_event.frames_visible >= self.min_frames_threshold:
                if search_event.outcome == 'searching':
                    search_event.outcome = 'exited_without_parking'
                    self.vehicles_failed_search += 1
                    self.logger.info(
                        f"Vehicle {track_id} exited without parking "
                        f"after {search_event.search_duration:.1f}s"
                    )

                self.completed_searches.append(search_event)

            del self.active_searches[track_id]

    def get_statistics(self) -> Dict:
        """Get aggregated search statistics"""
        if not self.completed_searches:
            return {
                'total_searches': 0,
                'successful_parks': self.vehicles_parked,
                'failed_searches': self.vehicles_failed_search,
                'success_rate': 0.0,
                'average_search_time': 0.0,
                'max_search_time': 0.0
            }

        search_times = [e.search_duration for e in self.completed_searches]
        total = self.vehicles_parked + self.vehicles_failed_search

        return {
            'total_searches': total,
            'successful_parks': self.vehicles_parked,
            'failed_searches': self.vehicles_failed_search,
            'success_rate': (self.vehicles_parked / total * 100) if total > 0 else 0.0,
            'average_search_time': sum(search_times) / len(search_times),
            'max_search_time': max(search_times),
            'min_search_time': min(search_times)
        }

    def get_search_events(self) -> List[Dict]:
        """Get all completed search events as dictionaries"""
        return [
            {
                'track_id': e.track_id,
                'entry_time': datetime.fromtimestamp(e.entry_time).isoformat(),
                'exit_time': datetime.fromtimestamp(e.exit_time).isoformat() if e.exit_time else None,
                'search_duration': e.search_duration,
                'frames_visible': e.frames_visible,
                'parked_spot_id': e.parked_spot_id,
                'outcome': e.outcome
            }
            for e in self.completed_searches
        ]

    def reset(self):
        """Reset analyzer state"""
        self.active_searches = {}
        self.completed_searches = []
        self.vehicles_parked = 0
        self.vehicles_failed_search = 0
