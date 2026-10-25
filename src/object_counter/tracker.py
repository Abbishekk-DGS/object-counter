"""SORT multi-object tracker, from scratch (Bewley et al., 2016).

Each track is a constant-velocity Kalman filter over [cx, cy, area, aspect].
Every frame: predict all tracks -> match to detections by IoU with the Hungarian
algorithm -> update matched tracks, start new ones, retire ones lost too long.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes in xyxy format. a: (N,4), b: (M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def xyxy_to_z(box: np.ndarray) -> np.ndarray:
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / max(h, 1e-6)])


def z_to_xyxy(x: np.ndarray) -> np.ndarray:
    area, ratio = max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(area * ratio)
    h = area / w
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class KalmanBox:
    """7-D state [cx, cy, s, r, vcx, vcy, vs]; aspect ratio r is assumed constant."""

    F = np.eye(7)
    F[0, 4] = F[1, 5] = F[2, 6] = 1.0
    H = np.eye(4, 7)

    def __init__(self, box: np.ndarray):
        self.x = np.zeros(7)
        self.x[:4] = xyxy_to_z(box)
        self.P = np.diag([10, 10, 10, 10, 1e4, 1e4, 1e4]).astype(float)  # unknown velocity
        self.Q = np.diag([1, 1, 1, 1, 0.01, 0.01, 1e-4])
        self.R = np.diag([1, 1, 10, 10])

    def predict(self) -> np.ndarray:
        if self.x[2] + self.x[6] <= 0:  # area can't go negative
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return z_to_xyxy(self.x)

    def update(self, box: np.ndarray) -> None:
        y = xyxy_to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        return z_to_xyxy(self.x)


@dataclass
class Track:
    id: int
    cls: int
    kf: KalmanBox
    hits: int = 1
    age: int = 0
    misses: int = 0
    last_obs: np.ndarray | None = None
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def box(self) -> np.ndarray:
        return self.kf.box

    @property
    def center(self) -> tuple[float, float]:
        b = self.box
        return float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)


class SORT:
    def __init__(self, max_misses: int = 15, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_misses, self.min_hits, self.iou_threshold = max_misses, min_hits, iou_threshold
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, boxes: np.ndarray, classes: np.ndarray | None = None) -> list[Track]:
        """boxes: (N,4) xyxy detections for this frame. Returns confirmed, currently-seen tracks."""
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        classes = np.zeros(len(boxes), int) if classes is None else np.asarray(classes, int)

        predicted = np.array([t.kf.predict() for t in self.tracks]).reshape(-1, 4)
        for t in self.tracks:
            t.age += 1

        ious = iou_matrix(predicted, boxes)
        # Never match across classes: a car can't become a person.
        if len(self.tracks) and len(boxes):
            same = np.array([t.cls for t in self.tracks])[:, None] == classes[None, :]
            ious = ious * same
        matched_t, matched_d = set(), set()

        def assign(scores: np.ndarray, t_idx: list[int], d_idx: list[int]) -> None:
            if not scores.size:
                return
            rows, cols = linear_sum_assignment(-scores)
            for r, c in zip(rows, cols, strict=True):
                if scores[r, c] >= self.iou_threshold:
                    ti, di = t_idx[r], d_idx[c]
                    track = self.tracks[ti]
                    track.kf.update(boxes[di])
                    track.last_obs = boxes[di]
                    track.hits += 1
                    track.misses = 0
                    track.history.append(track.center)
                    matched_t.add(ti)
                    matched_d.add(di)

        # Stage 1: detections vs Kalman predictions.
        assign(ious, list(range(len(self.tracks))), list(range(len(boxes))))

        # Stage 2 (observation-centric, as in OC-SORT): leftovers vs each track's last real
        # observation. Recovers objects that stop or reverse, where the prediction overshoots.
        t_left = [i for i in range(len(self.tracks)) if i not in matched_t]
        d_left = [j for j in range(len(boxes)) if j not in matched_d]
        if t_left and d_left:
            last = np.array([self.tracks[i].last_obs for i in t_left])
            ious2 = iou_matrix(last, boxes[d_left])
            ious2 *= (
                np.array([self.tracks[i].cls for i in t_left])[:, None] == classes[d_left][None, :]
            )
            assign(ious2, t_left, d_left)

        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.misses += 1
        for j in range(len(boxes)):
            if j not in matched_d:
                track = Track(
                    self._next_id, int(classes[j]), KalmanBox(boxes[j]), last_obs=boxes[j]
                )
                track.history.append(track.center)
                self.tracks.append(track)
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return [t for t in self.tracks if t.misses == 0 and t.hits >= self.min_hits]
