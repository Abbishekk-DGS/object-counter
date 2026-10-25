import numpy as np

from object_counter.analytics import Heatmap, LineCounter, Zone, point_in_polygon
from object_counter.tracker import SORT, iou_matrix, xyxy_to_z, z_to_xyxy

NAMES = {0: "person", 2: "car"}


def box(cx, cy, w=20, h=40):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def test_iou_basics():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
    assert np.allclose(iou_matrix(a, b)[0], [1.0, 25 / 175, 0.0])


def test_box_state_roundtrip():
    b = np.array([10.0, 20.0, 50.0, 100.0])
    assert np.allclose(z_to_xyxy(xyxy_to_z(b)), b)


def test_ids_stay_stable_for_crossing_objects():
    """Two people walk towards each other on parallel lanes; IDs must not swap."""
    sort = SORT(min_hits=1)
    for f in range(30):
        tracks = sort.update(np.array([box(10 + 5 * f, 100), box(160 - 5 * f, 160)]))
    by_lane = {round(t.center[1] / 10): t.id for t in tracks}
    assert by_lane == {10: 1, 16: 2}


def test_track_survives_short_occlusion():
    sort = SORT(min_hits=1, max_misses=5)
    for f in range(10):
        sort.update(np.array([box(10 + 4 * f, 50)]))
    for _ in range(3):  # object hidden for 3 frames
        sort.update(np.zeros((0, 4)))
    tracks = sort.update(np.array([box(10 + 4 * 13, 50)]))
    assert [t.id for t in tracks] == [1]  # Kalman prediction carried it through


def test_min_hits_suppresses_flicker():
    sort = SORT(min_hits=3)
    assert sort.update(np.array([box(50, 50)])) == []
    sort.update(np.array([box(52, 50)]))
    assert len(sort.update(np.array([box(54, 50)]))) == 1


def test_classes_never_merge():
    sort = SORT(min_hits=1)
    sort.update(np.array([box(50, 50)]), np.array([0]))
    tracks = sort.update(np.array([box(51, 50)]), np.array([2]))
    assert {t.id for t in sort.tracks} == {1, 2} and tracks[0].cls == 2


def test_line_counting_ignores_jitter_and_reversals():
    """A walks down across the line, then abruptly jitters back toward it. The prediction
    overshoots; the observation-centric stage keeps A's ID so it is not counted twice."""
    sort = SORT(min_hits=1)
    line = LineCounter((0, 100), (400, 100))
    ys_a = [60, 75, 90, 105, 98, 110, 125, 140]
    ys_b = [180, 160, 140, 120, 85, 70, 50, 30]
    for f, (ya, yb) in enumerate(zip(ys_a, ys_b, strict=True)):
        tracks = sort.update(np.array([box(100, ya), box(300, yb)]), np.array([0, 2]))
        line.update(tracks, f, NAMES)
    assert {t.id for t in sort.tracks} == {1, 2}  # no ID switch
    assert line.totals() == {"in": 1, "out": 1}
    assert [(e["class"], e["direction"]) for e in line.events] == [("person", "in"), ("car", "out")]


def test_each_track_counted_once_per_direction():
    sort = SORT(min_hits=1)
    line = LineCounter((0, 100), (400, 100))
    path = [40, 60, 80, 96, 112, 128, 140, 146, 140, 128, 112, 96, 80, 64, 56, 64, 80, 96, 112, 128]
    for f, y in enumerate(path):  # down, back up, down again: smooth turns
        line.update(sort.update(np.array([box(100, y)])), f, NAMES)
    assert line.totals() == {"in": 1, "out": 1}


def test_zone_dwell_and_heatmap():
    square = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert point_in_polygon((50, 50), square) and not point_in_polygon((150, 50), square)
    sort = SORT(min_hits=1)
    zone = Zone(square, fps=10)
    heat = Heatmap(200, 200, cell=50)
    for _ in range(20):
        tracks = sort.update(np.array([box(50, 50), box(150, 150)]))
        zone.update(tracks)
        heat.update(tracks)
    assert zone.dwell_seconds() == {1: 2.0}
    assert heat.normalized().max() == 1.0 and heat.grid[1, 1] == 20
