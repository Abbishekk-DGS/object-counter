"""object-counter: count people, cars or anything YOLO can see crossing a line.

  object-counter traffic.mp4 --line 0,0.6,1,0.6 --classes car,truck,bus,motorcycle
  object-counter 0 --line 0.5,0,0.5,1 --classes person            # webcam
  object-counter shop.mp4 --zone 0.1,0.5,0.4,0.5,0.4,0.9,0.1,0.9  # dwell time in a polygon

Coordinates are fractions of the frame (0..1) so the same flags work at any resolution.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .analytics import Heatmap, LineCounter, Zone
from .tracker import SORT

PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
           (10, 249, 72), (23, 204, 146), (134, 219, 61), (211, 188, 0), (209, 99, 0)]  # fmt: skip


def parse_points(spec: str, w: int, h: int) -> list[tuple[float, float]]:
    vals = [float(v) for v in spec.split(",")]
    if len(vals) % 2:
        raise ValueError("points need an even number of values")
    return [(vals[i] * w, vals[i + 1] * h) for i in range(0, len(vals), 2)]


def draw(frame, tracks, names, line, zone, heat, inside):
    if heat is not None:
        overlay = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        overlay = cv2.resize(
            overlay, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC
        )
        frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)
    if zone:
        cv2.polylines(frame, [np.int32(zone.polygon)], True, (0, 200, 255), 2)
    if line:
        cv2.line(frame, tuple(map(int, line.a)), tuple(map(int, line.b)), (255, 255, 255), 2)
    for t in tracks:
        color = PALETTE[t.id % len(PALETTE)]
        x1, y1, x2, y2 = map(int, t.box)
        thick = 3 if t.id in inside else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        label = f"#{t.id} {names.get(t.cls, t.cls)}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        pts = np.int32(t.history[-30:])
        if len(pts) > 1:
            cv2.polylines(frame, [pts], False, color, 2)
    if line:
        tot = line.totals()
        text = f"IN {tot.get('in', 0)}   OUT {tot.get('out', 0)}"
        cv2.rectangle(frame, (10, 10), (240, 48), (0, 0, 0), -1)
        cv2.putText(frame, text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    h = frame.shape[0]
    cv2.putText(
        frame,
        "object-counter - Abbishekk DGS",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
    )
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="object-counter",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} · by Abbishekk DGS · abbishekkdgs.tech",
    )
    parser.add_argument("source", help="video file, image folder, RTSP URL, or webcam index")
    parser.add_argument("--model", default="yolo11n.pt", help="any Ultralytics detection model")
    parser.add_argument("--classes", default="person", help="comma-separated COCO class names")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--line", default="0,0.5,1,0.5", help="x1,y1,x2,y2 as frame fractions")
    parser.add_argument("--zone", default=None, help="polygon x1,y1,x2,y2,... as frame fractions")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--show", action="store_true", help="live preview window")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(args.model)
    names: dict[int, str] = model.names
    wanted = {n.strip() for n in args.classes.split(",")}
    class_ids = [i for i, n in names.items() if n in wanted]
    if not class_ids:
        parser.error(f"none of {wanted} are classes of {args.model}")

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not w:
        raise SystemExit(f"could not open {args.source}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(str(args.source)).stem or "webcam"
    writer = cv2.VideoWriter(
        str(out_dir / f"{stem}_counted.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    tracker = SORT(max_misses=int(fps // 2), min_hits=3)
    a, b = parse_points(args.line, w, h)
    line = LineCounter(a, b)
    zone = Zone(parse_points(args.zone, w, h), fps) if args.zone else None
    heat = Heatmap(w, h) if args.heatmap else None

    frame_i, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, conf=args.conf, classes=class_ids, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        cls = res.boxes.cls.cpu().numpy().astype(int) if len(res.boxes) else np.zeros(0, int)
        tracks = tracker.update(boxes, cls)
        line.update(tracks, frame_i, names)
        inside = zone.update(tracks) if zone else []
        if heat:
            heat.update(tracks)
        annotated = draw(
            frame, tracks, names, line, zone, heat.normalized() if heat else None, inside
        )
        writer.write(annotated)
        if args.show:
            cv2.imshow("object-counter", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_i += 1
        if frame_i % 50 == 0:
            print(f"frame {frame_i} | {frame_i / (time.time() - t0):.1f} fps | {line.totals()}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    with (out_dir / f"{stem}_events.csv").open("w", newline="") as f:
        fields = ["frame", "line", "track", "class", "direction"]
        csv.DictWriter(f, fieldnames=fields).writeheader()
        csv.DictWriter(f, fieldnames=fields).writerows(line.events)
    summary = {
        "frames": frame_i,
        "totals": line.totals(),
        "by_class": {f"{d}:{c}": n for (d, c), n in line.counts.items()},
        "unique_tracks": tracker._next_id - 1,
    }
    if zone:
        dwell = zone.dwell_seconds()
        summary["zone"] = {
            "objects": len(dwell),
            "avg_dwell_s": round(sum(dwell.values()) / len(dwell), 2) if dwell else 0,
            "max_dwell_s": round(max(dwell.values()), 2) if dwell else 0,
        }
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
