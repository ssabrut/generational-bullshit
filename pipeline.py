from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import networkx as nx
import numpy as np
from skimage import img_as_bool
from skimage.morphology import skeletonize
from ultralytics import YOLO

from preprocessing import FloorPlanPreprocessor, PipelineResult

OPENING_CLASSES = {"door", "2door", "window"}
SNAP_PX = 20  # bbox proximity (px) to count a corner as "touching" an opening


@dataclass
class SkeletonGraph:
    skeleton_raw: np.ndarray        # 1-px binary skeleton
    skeleton_thick: np.ndarray      # dilated skeleton used for corner detection
    corners: list[tuple[int, int]]  # NMS-filtered corners snapped onto skeleton
    graph: nx.Graph                 # undirected wall graph (nodes=corners, edges=walls)


@dataclass
class FloorPlanOutput:
    prep: PipelineResult
    skel: SkeletonGraph
    graph: nx.Graph                 # combined graph: corner + opening nodes
    json: dict[str, Any]            # structured JSON ready for export


# ── Binary cleanup helpers ────────────────────────────────────────────────────

def _remove_noise(binary: np.ndarray, min_area_ratio: float = 0.0001) -> np.ndarray:
    h, w = binary.shape
    min_area = max(50, int(h * w * min_area_ratio))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 255
    return cleaned


def _morphological_cleanup(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if kernel_size <= 0:
        return binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)


# ── Corner detection helpers ──────────────────────────────────────────────────

def _detect_corners_polygon(
    binary: np.ndarray,
    min_area: int = 100,
    epsilon_ratio: float = 0.001,
) -> list[tuple[int, int]]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    corners: list[tuple[int, int]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        epsilon = epsilon_ratio * cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        for pt in approx:
            corners.append(tuple(pt[0]))
    return corners


def _suppress_neighbors(corners: list[tuple[int, int]], radius: int = 25) -> list[tuple[int, int]]:
    kept: list[tuple[int, int]] = []
    suppressed: set[int] = set()
    for i, (x0, y0) in enumerate(corners):
        if i in suppressed:
            continue
        kept.append((x0, y0))
        for j, (x1, y1) in enumerate(corners):
            if j != i and j not in suppressed:
                if (x1 - x0) ** 2 + (y1 - y0) ** 2 <= radius ** 2:
                    suppressed.add(j)
    return kept


def _snap_to_skeleton(
    corners: list[tuple[int, int]],
    skeleton_1px: np.ndarray,
    search_radius: int = 10,
) -> list[tuple[int, int]]:
    H, W = skeleton_1px.shape
    snapped: list[tuple[int, int]] = []
    for (x, y) in corners:
        x, y = int(x), int(y)
        x0, x1 = max(0, x - search_radius), min(W, x + search_radius + 1)
        y0, y1 = max(0, y - search_radius), min(H, y + search_radius + 1)
        window = skeleton_1px[y0:y1, x0:x1]
        ys, xs = np.where(window > 0)
        if len(xs) == 0:
            continue
        dx, dy = xs + x0 - x, ys + y0 - y
        idx = np.argmin(dx * dx + dy * dy)
        snapped.append((int(xs[idx] + x0), int(ys[idx] + y0)))
    return snapped


# ── BFS edge discovery ────────────────────────────────────────────────────────

def _bfs_edges(
    corners: list[tuple[int, int]],
    skeleton_1px: np.ndarray,
) -> dict[frozenset, int]:
    H, W = skeleton_1px.shape
    labels = -np.ones((H, W), dtype=np.int32)
    dist = -np.ones((H, W), dtype=np.int32)

    q: deque[tuple[int, int]] = deque()
    for cid, (x, y) in enumerate(corners):
        if labels[y, x] == -1:
            labels[y, x] = cid
            dist[y, x] = 0
            q.append((y, x))

    edges: dict[frozenset, int] = {}
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while q:
        y, x = q.popleft()
        my_label, my_dist = labels[y, x], dist[y, x]
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W or skeleton_1px[ny, nx] == 0:
                continue
            if labels[ny, nx] == -1:
                labels[ny, nx] = my_label
                dist[ny, nx] = my_dist + 1
                q.append((ny, nx))
            elif labels[ny, nx] != my_label:
                key = frozenset({my_label, int(labels[ny, nx])})
                length = my_dist + dist[ny, nx] + 1
                if key not in edges or length < edges[key]:
                    edges[key] = length

    return edges


# ── YOLO detection ───────────────────────────────────────────────────────────

def detect_openings(
    model: YOLO,
    color_bgr: np.ndarray,
    conf: float = 0.25,
    imgsz: int = 1024,
) -> list[dict[str, Any]]:
    """
    Run YOLO on a cropped BGR color image.
    Returns one dict per detection: {label, cx, cy, bbox (x1,y1,x2,y2), conf}.
    Coordinates are in the same cropped space as the skeleton graph.
    Only classes in OPENING_CLASSES are returned.
    """
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    results = model(rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
    openings: list[dict[str, Any]] = []
    for box in results.boxes:
        label = model.names[int(box.cls)]
        if label not in OPENING_CLASSES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        openings.append(dict(
            label=label,
            cx=(x1 + x2) / 2,
            cy=(y1 + y2) / 2,
            bbox=(x1, y1, x2, y2),
            conf=float(box.conf),
        ))
    return openings


def _gap_corners(G: nx.Graph, x1: float, y1: float, x2: float, y2: float) -> list[int]:
    """
    For each side of the bbox, find corner nodes within SNAP_PX.
    When multiple candidates exist on the same side, keep the one closest
    to that side's edge (left-side → smallest x; top-side → smallest y;
    right-side → largest x; bottom-side → largest y).
    Returns at most one node-id per side (up to 4 total, deduped).
    """
    sides: list[tuple[str, list[tuple[float, int]]]] = [
        ("left",   []),
        ("right",  []),
        ("top",    []),
        ("bottom", []),
    ]

    for n, attr in G.nodes(data=True):
        if attr["kind"] != "corner":
            continue
        cx, cy = attr["x"], attr["y"]

        if (x1 - SNAP_PX) <= cx <= (x1 + SNAP_PX) and (y1 - SNAP_PX) <= cy <= (y2 + SNAP_PX):
            sides[0][1].append((cx, n))          # left: sort ascending → closest to x1
        if (x2 - SNAP_PX) <= cx <= (x2 + SNAP_PX) and (y1 - SNAP_PX) <= cy <= (y2 + SNAP_PX):
            sides[1][1].append((-cx, n))         # right: sort ascending → closest to x2 (largest cx)
        if (y1 - SNAP_PX) <= cy <= (y1 + SNAP_PX) and (x1 - SNAP_PX) <= cx <= (x2 + SNAP_PX):
            sides[2][1].append((cy, n))          # top: sort ascending → closest to y1
        if (y2 - SNAP_PX) <= cy <= (y2 + SNAP_PX) and (x1 - SNAP_PX) <= cx <= (x2 + SNAP_PX):
            sides[3][1].append((-cy, n))         # bottom: sort ascending → closest to y2 (largest cy)

    selected: list[int] = []
    seen: set[int] = set()
    for _, candidates in sides:
        if candidates:
            candidates.sort()
            nid = candidates[0][1]
            if nid not in seen:
                selected.append(nid)
                seen.add(nid)
    return selected


def attach_openings(G: nx.Graph, openings: list[dict[str, Any]]) -> nx.Graph:
    """
    Add each opening as a new node and connect it to the wall-gap corners.

    For each side of the bbox, the corner closest to that edge is selected
    (so up to 2 corners for a straight door, one on each wall endpoint).
    Edges span from those corners through the opening center, filling the gap.
    Falls back to nearest corner if no corners are found near the bbox.
    Mutates G in-place and returns it.
    """
    next_id = max(G.nodes()) + 1 if G.nodes() else 0
    for det in openings:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = det["cx"], det["cy"]

        gap = _gap_corners(G, x1, y1, x2, y2)

        if not gap:
            # fallback: nearest corner by Euclidean distance from center
            best_node, best_d2 = None, float("inf")
            for n, attr in G.nodes(data=True):
                if attr["kind"] != "corner":
                    continue
                d2 = (attr["x"] - cx) ** 2 + (attr["y"] - cy) ** 2
                if d2 < best_d2:
                    best_d2, best_node = d2, n
            gap = [best_node] if best_node is not None else []

        G.add_node(next_id, x=cx, y=cy, kind=det["label"], bbox=det["bbox"], conf=det["conf"])
        for nid in gap:
            attr = G.nodes[nid]
            dist = ((attr["x"] - cx) ** 2 + (attr["y"] - cy) ** 2) ** 0.5
            G.add_edge(next_id, nid, length=dist, kind="opening")
        next_id += 1
    return G


# ── JSON export ───────────────────────────────────────────────────────────────

def graph_to_json(G: nx.Graph) -> dict[str, Any]:
    """
    Convert the combined graph to the structured output JSON.

    Schema
    ------
    {
      "walls":   [{"from":[x,y], "to":[x,y], "length":px}, ...],
      "doors":   [{"center":[x,y], "bbox":[x1,y1,x2,y2],
                   "conf":0.9, "connected_to":[[x,y],...],
                   "touching_corners":[[x,y],...]},...],
      "windows": [...same as doors...]
    }
    """
    walls: list[dict] = []
    doors: list[dict] = []
    windows: list[dict] = []

    for a, b, edata in G.edges(data=True):
        if edata["kind"] == "wall":
            walls.append({
                "from":   [G.nodes[a]["x"], G.nodes[a]["y"]],
                "to":     [G.nodes[b]["x"], G.nodes[b]["y"]],
                "length": edata["length"],
            })

    for n, attr in G.nodes(data=True):
        if attr["kind"] == "corner":
            continue

        nbrs = [nb for nb in G.neighbors(n) if G.nodes[nb]["kind"] == "corner"]
        connected_to = [[G.nodes[nb]["x"], G.nodes[nb]["y"]] for nb in nbrs]

        x1, y1, x2, y2 = attr["bbox"]
        touching = [
            [ca["x"], ca["y"]]
            for _, ca in G.nodes(data=True)
            if ca["kind"] == "corner"
            and (x1 - SNAP_PX) <= ca["x"] <= (x2 + SNAP_PX)
            and (y1 - SNAP_PX) <= ca["y"] <= (y2 + SNAP_PX)
        ]

        entry = {
            "center":            [round(attr["x"], 1), round(attr["y"], 1)],
            "bbox":              [round(v, 1) for v in (x1, y1, x2, y2)],
            "conf":              round(attr["conf"], 3),
            "connected_to":      connected_to,
            "touching_corners":  touching,
        }

        if attr["kind"] in ("door", "2door"):
            doors.append(entry)
        elif attr["kind"] == "window":
            windows.append(entry)

    return {"walls": walls, "doors": doors, "windows": windows}


# ── Public API ────────────────────────────────────────────────────────────────

def build_skeleton_graph(
    preprocessed: np.ndarray,
    dilate_size: int = 3,
    min_area: int = 100,
    epsilon_ratio: float = 0.001,
    nms_radius: int = 25,
    snap_radius: int = 10,
) -> SkeletonGraph:
    """
    Build a wall graph from a binary (cropped, preprocessed) floor plan image.

    Parameters
    ----------
    preprocessed : binary np.ndarray — output of FloorPlanPreprocessor.process()
    dilate_size  : kernel size for thickening the 1-px skeleton before corner detection
    min_area     : minimum contour area to consider during corner detection
    epsilon_ratio: polygon approximation tolerance (smaller = more corners)
    nms_radius   : non-maximum suppression radius in pixels
    snap_radius  : search window for snapping corners onto the 1-px skeleton

    Returns
    -------
    SkeletonGraph with skeleton_raw, skeleton_thick, corners, and graph
    """
    # Preprocessor already crops to content — skip the redundant second crop
    # to keep skeleton coordinates in the same space as raw_no_text (for YOLO).
    binary = cv2.threshold(preprocessed, 127, 255, cv2.THRESH_BINARY)[1]
    cleaned = _remove_noise(binary, min_area_ratio=0.0001)
    smoothed = _morphological_cleanup(cleaned, kernel_size=3)

    # Skeletonize
    skeleton_raw = (skeletonize(img_as_bool(smoothed)) * 255).astype(np.uint8)

    # Thicken for polygon corner detection
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size))
    skeleton_thick = cv2.dilate(skeleton_raw, k)

    # Corner detection → NMS → snap onto 1-px skeleton
    poly_corners = _detect_corners_polygon(skeleton_thick, min_area=min_area, epsilon_ratio=epsilon_ratio)
    kept_corners = _suppress_neighbors(poly_corners, radius=nms_radius)
    snapped = _snap_to_skeleton(kept_corners, skeleton_raw, search_radius=snap_radius)

    # BFS edge discovery → NetworkX graph
    edge_dict = _bfs_edges(snapped, skeleton_raw)
    G = nx.Graph()
    for cid, (x, y) in enumerate(snapped):
        G.add_node(cid, x=x, y=y, kind="corner")
    for key, length in edge_dict.items():
        a, b = tuple(key)
        G.add_edge(a, b, length=int(length), kind="wall")

    return SkeletonGraph(
        skeleton_raw=skeleton_raw,
        skeleton_thick=skeleton_thick,
        corners=snapped,
        graph=G,
    )


def run_pipeline(
    image: np.ndarray,
    model_path: str | Path = "Models/best_v2.pt",
    preprocessor: FloorPlanPreprocessor | None = None,
    conf: float = 0.25,
    imgsz: int = 1024,
) -> FloorPlanOutput:
    """
    Run the full pipeline on a single BGR floor plan image.

    Steps
    -----
    1–3  Text removal, binarisation, crop  (FloorPlanPreprocessor)
    4    Skeleton → corner graph            (build_skeleton_graph)
    5    YOLO door/window detection         (detect_openings)
    6    Graph fusion + JSON export         (attach_openings, graph_to_json)

    Parameters
    ----------
    image      : BGR numpy array
    model_path : path to YOLO weights (.pt)
    preprocessor : optional pre-instantiated FloorPlanPreprocessor
    conf       : YOLO confidence threshold
    imgsz      : YOLO inference size

    Returns
    -------
    FloorPlanOutput with .prep, .skel, .graph, and .json
    """
    if preprocessor is None:
        preprocessor = FloorPlanPreprocessor(angle_step=90)

    prep = preprocessor.process(image)
    skel = build_skeleton_graph(prep.preprocessed)

    model = YOLO(str(model_path))
    openings = detect_openings(model, prep.raw_no_text, conf=conf, imgsz=imgsz)

    # Work on a copy so skel.graph stays pure (corners + walls only)
    combined = skel.graph.copy()
    attach_openings(combined, openings)

    return FloorPlanOutput(
        prep=prep,
        skel=skel,
        graph=combined,
        json=graph_to_json(combined),
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    img_path   = sys.argv[1] if len(sys.argv) > 1 else "data/PNG/raw/IIa_A01001.png"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "Models/best_v2.pt"

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(img_path)

    out = run_pipeline(image, model_path=model_path)
    G   = out.graph

    print(f"rotation_angle : {out.prep.rotation_angle}°")
    print(f"crop_bbox      : {out.prep.crop_bbox}")
    print(f"corners        : {len(out.skel.corners)}")
    print(f"walls          : {len(out.json['walls'])}")
    print(f"doors          : {len(out.json['doors'])}")
    print(f"windows        : {len(out.json['windows'])}")
    print(f"graph          : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    out_dir = Path("data/PNG")
    stem    = Path(img_path).stem

    (out_dir / "no_text").mkdir(parents=True, exist_ok=True)
    (out_dir / "crop_and_pad").mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "no_text"      / f"{stem}.png"), out.prep.raw_no_text)
    cv2.imwrite(str(out_dir / "crop_and_pad" / f"{stem}.png"), out.prep.preprocessed)

    # Overlay: skeleton + wall edges + opening bboxes
    vis = cv2.cvtColor(out.skel.skeleton_raw, cv2.COLOR_GRAY2BGR)
    COLORS = {"wall": (0, 255, 0), "opening": (0, 165, 255)}
    for a, b, edata in G.edges(data=True):
        pa = (int(G.nodes[a]["x"]), int(G.nodes[a]["y"]))
        pb = (int(G.nodes[b]["x"]), int(G.nodes[b]["y"]))
        cv2.line(vis, pa, pb, COLORS.get(edata["kind"], (200, 200, 200)), 2)
    for n, attr in G.nodes(data=True):
        p = (int(attr["x"]), int(attr["y"]))
        color = (0, 0, 255) if attr["kind"] == "corner" else (0, 165, 255)
        cv2.circle(vis, p, 6, color, -1)
        cv2.circle(vis, p, 6, (255, 255, 255), 1)
        if "bbox" in attr:
            x1, y1, x2, y2 = (int(v) for v in attr["bbox"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(vis, f"{attr['kind']} {attr['conf']:.2f}",
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    overlay_path = str(out_dir / f"{stem}_graph.png")
    cv2.imwrite(overlay_path, vis)
    print(f"Saved overlay  → {overlay_path}")

    json_path = str(out_dir / f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(out.json, f, indent=2)
    print(f"Saved JSON     → {json_path}")
    print(json.dumps(out.json, indent=2))
