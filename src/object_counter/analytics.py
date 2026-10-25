"""Turn tracks into numbers: directional line crossings, zone dwell time, motion heatmap."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .tracker import Track

Point = tuple[float, float]


def _side(a: Point, b: Point, p: Point) -> float:
    """Sign of the cross product: which side of line a->b point p is on."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _segments_intersect(p1: Point, p2: Point, a: Point, b: Point) -> bool:
    d1, d2 = _side(a, b, p1), _side(a, b, p2)
    d3, d4 = _side(p1, p2, a), _side(p1, p2, b)
    return d1 * d2 < 0 and d3 * d4 < 0


@dataclass
class LineCounter:
    """Counts each track at most once per direction when its path crosses the segment a-b."""

    a: Point
    b: Point
    name: str = "line"
    counts: Counter = field(default_factory=Counter)  # (direction, class) -> n
    _counted: set = field(default_factory=set)
    events: list[dict] = field(default_factory=list)

    def update(self, tracks: list[Track], frame: int, names: dict[int, str]) -> None:
        for t in tracks:
            if len(t.history) < 2:
                continue
            prev, cur = t.history[-2], t.history[-1]
            if not _segments_intersect(prev, cur, self.a, self.b):
                continue
            direction = "in" if _side(self.a, self.b, cur) > 0 else "out"
            key = (t.id, direction)
            if key in self._counted:
                continue
            self._counted.add(key)
            label = names.get(t.cls, str(t.cls))
            self.counts[(direction, label)] += 1
            self.events.append(
                {
                    "frame": frame,
                    "line": self.name,
                    "track": t.id,
                    "class": label,
                    "direction": direction,
                }
            )

    def totals(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for (direction, _), n in self.counts.items():
            out[direction] += n
        return dict(out)


def point_in_polygon(p: Point, poly: list[Point]) -> bool:
    """Ray casting."""
    inside = False
    x, y = p
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


@dataclass
class Zone:
    """Tracks how long each object stays inside a polygon (queues, restricted areas)."""

    polygon: list[Point]
    fps: float
    name: str = "zone"
    frames_inside: Counter = field(default_factory=Counter)

    def update(self, tracks: list[Track]) -> list[int]:
        inside = [t.id for t in tracks if point_in_polygon(t.center, self.polygon)]
        self.frames_inside.update(inside)
        return inside

    def dwell_seconds(self) -> dict[int, float]:
        return {tid: n / self.fps for tid, n in self.frames_inside.items()}


class Heatmap:
    def __init__(self, width: int, height: int, cell: int = 16, decay: float = 1.0):
        self.cell, self.decay = cell, decay
        self.grid = np.zeros((height // cell + 1, width // cell + 1), np.float32)

    def update(self, tracks: list[Track]) -> None:
        self.grid *= self.decay
        for t in tracks:
            cx, cy = t.center
            gy = int(np.clip(cy // self.cell, 0, self.grid.shape[0] - 1))
            gx = int(np.clip(cx // self.cell, 0, self.grid.shape[1] - 1))
            self.grid[gy, gx] += 1

    def normalized(self) -> np.ndarray:
        peak = self.grid.max()
        return self.grid / peak if peak > 0 else self.grid
