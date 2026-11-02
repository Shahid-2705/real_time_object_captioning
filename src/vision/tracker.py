"""Lightweight object tracking abstractions and utilities."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class CentroidTracker:
    """Lightweight Euclidean-distance centroid tracker for bounding boxes."""

    def __init__(self, max_disappeared: int = 15, max_distance: float = 80.0) -> None:
        self.next_object_id = 0
        self.objects: Dict[int, np.ndarray] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: np.ndarray) -> int:
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.disappeared[obj_id] = 0
        self.next_object_id += 1
        return obj_id

    def deregister(self, object_id: int) -> None:
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, rects: List[Tuple[int, int, int, int]]) -> Dict[int, np.ndarray]:
        """Update tracker with list of (x1, y1, x2, y2) bounding boxes."""
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            # Compute Euclidean distances between existing objects and input centroids
            distances = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)

            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if distances[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, distances.shape[0])).difference(used_rows)
            unused_cols = set(range(0, distances.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects
