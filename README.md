# object-counter

**Count people, vehicles or anything YOLO can see, from any video, webcam or RTSP stream.** Detection uses Ultralytics YOLO11, and the multi-object tracker is **written from scratch** here: a Kalman filter per object, Hungarian matching, plus an observation-centric recovery stage. On top of that sit directional line counting, zone dwell time and motion heatmaps.

![python](https://img.shields.io/badge/python-3.11+-blue) ![license](https://img.shields.io/badge/license-AGPL--3.0-blue) ![runs](https://img.shields.io/badge/runs-offline-black)

<!-- author-banner -->
> **Project 06 of 20** in my AI build series, designed and built by **[Abbishekk DGS](https://abbishekkdgs.tech)**. [See all projects →](https://github.com/Abbishekk-DGS)

<!-- overview:start -->
## Overview

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/overview-dark.svg">
  <img alt="object-counter overview: how it works and its output. Key numbers: ~23 fps end to end, 8 GB Mac; 596 frames in the test video; 12 unique tracks; 3 line events: 2 in, 1 out. Stack: Python · Ultralytics YOLO11 · Kalman filter · Hungarian matching · OpenCV." src="docs/overview-light.svg" width="100%">
</picture>

## Interface

It is one terminal command, with an optional live preview window.

```
object-counter <video | RTSP URL | webcam index> [options]
```

| Option | What it does | Default |
|---|---|---|
| `--classes` | COCO class names to count, comma separated | `person` |
| `--line` | Counting line as `x1,y1,x2,y2` frame fractions | `0,0.5,1,0.5` |
| `--zone` | Polygon for dwell time, as frame fractions | |
| `--heatmap` | Adds the motion heatmap | off |
| `--model` | Any Ultralytics detection model | `yolo11n.pt` |
| `--conf` | Detection confidence | 0.35 |
| `--show` | Opens a live preview; press `q` to stop | off |
| `--out` | Output folder | `outputs` |

The terminal prints fps and running totals every 50 frames. Each run writes `<name>_counted.mp4`, `<name>_events.csv` (frame, line, track, class, direction) and `<name>_summary.json`, and prints the summary at the end.
<!-- overview:end -->

## What you get

- **Directional line counting**: IN / OUT per class. Each object is counted at most once per direction, so people lingering on the line don't inflate numbers
- **Zone analytics**: dwell time per object inside any polygon (queues, counters, restricted areas)
- **Motion heatmap** overlay
- **Outputs**: annotated video, `events.csv` (frame, track, class, direction), `summary.json`
- Resolution-independent config: lines and zones are given as frame fractions
- Works with any Ultralytics model (`yolo11n` → `yolo11x`, custom-trained weights)

## Results

[Intel sample video](https://github.com/intel-iot-devkit/sample-videos) `people-detection.mp4` (768×432, 596 frames), YOLO11n, on an 8 GB Apple Silicon Mac:

| Metric | Value |
|---|---|
| Throughput | **~23 fps** end-to-end (detect + track + annotate + encode) |
| Line events | 3 (1 out, 2 in) |
| Unique tracks | 12 |

## Quickstart

```bash
git clone https://github.com/Abbishekk-DGS/object-counter && cd object-counter
uv sync

# people crossing a vertical line through the middle of the frame
uv run object-counter mall.mp4 --classes person --line 0.5,0,0.5,1

# traffic: vehicles crossing a horizontal line, with heatmap
uv run object-counter traffic.mp4 --classes car,truck,bus,motorcycle --line 0,0.6,1,0.6 --heatmap

# dwell time inside a polygon, live webcam preview
uv run object-counter 0 --zone 0.1,0.5,0.4,0.5,0.4,0.9,0.1,0.9 --show
```

The first run downloads `yolo11n.pt` (~6 MB).

## How the tracker works

```mermaid
flowchart LR
    D[YOLO detections<br/>boxes + classes] --> A1
    P[Kalman predict<br/>every track] --> A1[Stage 1: Hungarian on IoU<br/>detections × predictions]
    A1 -->|unmatched| A2[Stage 2: Hungarian on IoU<br/>detections × last observations]
    A1 -->|matched| U[Kalman update]
    A2 -->|matched| U
    A2 -->|unmatched detections| N[new tracks]
    A2 -->|unmatched tracks| M[miss++ → retire after N]
    U --> C[line / zone / heatmap analytics]
```

- **State**: `[cx, cy, area, aspect, vx, vy, v_area]` with a constant-velocity model ([SORT](https://arxiv.org/abs/1602.00763)).
- **Class-gated matching**: a car can never inherit a person's ID.
- **Observation-centric recovery** ([OC-SORT](https://arxiv.org/abs/2203.14360) idea): when an object stops or reverses, the Kalman prediction overshoots and the IoU match fails. A second pass matches leftovers against each track's *last real box*. The test suite includes this exact case: without stage 2 the walker gets a new ID and is counted twice.
- **Crossing test**: a segment-intersection check between the line and the track's movement each frame, using cross-product signs. It's exact and needs no "line zone" heuristics.

## Tests

```bash
uv sync --extra dev && uv run pytest -q
```

Tests are synthetic and need no model: stable IDs for objects passing each other, survival through a 3-frame occlusion, class gating, no double-count on reversal, once-per-direction counting, zone dwell time and heatmap accumulation.

## About the author

Built by **Abbishekk DGS**. More of my work at **[abbishekkdgs.tech](https://abbishekkdgs.tech)** and on **[GitHub](https://github.com/Abbishekk-DGS)**.

## License

AGPL-3.0, to match the Ultralytics YOLO dependency. The tracker and analytics modules (`tracker.py`, `analytics.py`) are self-contained and only depend on numpy and scipy.
