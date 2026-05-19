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
from window_detect import detect_windows_fm

OPENING_CLASSES = {"door", "2door", "window"}
SNAP_PX = 40  # search radius beyond the bbox edge for gap-corner candidates


@dataclass
class SkeletonGraph:
    skeleton_raw: np.ndarray        # 1-px binary skeleton (after spur pruning)
    skeleton_thick: np.ndarray      # dilated skeleton for visualisation
    corners: list[tuple[int, int]]  # topology corners after NMS
    graph: nx.Graph                 # undirected wall graph (nodes=corners, edges=walls)
    dag: nx.DiGraph                 # DFS spanning-forest DAG of the wall graph


@dataclass
class FloorPlanOutput:
    prep: PipelineResult
    skel: SkeletonGraph
    graph: nx.Graph                 # combined graph: corner + opening nodes
    json: dict[str, Any]            # structured JSON ready for export


# ── Skeleton helpers ──────────────────────────────────────────────────────────

def _prune_spurs(skeleton: np.ndarray, iterations: int = 20) -> np.ndarray:
    """Iteratively remove 1-neighbor skeleton pixels (dangling tips)."""
    skel   = (skeleton > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(iterations):
        n_count   = cv2.filter2D(skel, -1, kernel) - skel
        endpoints = (skel == 1) & (n_count == 1)
        if not endpoints.any():
            break
        skel[endpoints] = 0
    return (skel * 255).astype(np.uint8)


def _detect_corners_topology(skel_1px: np.ndarray) -> list[tuple[int, int]]:
    """Return endpoints, junctions, and non-collinear bends on the skeleton."""
    skel = (skel_1px > 0).astype(np.uint8)
    n    = cv2.filter2D(skel, -1, np.ones((3, 3), np.uint8)) - skel
    endpoints = (skel == 1) & (n == 1)
    junctions = (skel == 1) & (n >= 3)
    straight  = (
        (cv2.filter2D(skel, -1, np.array([[0,0,0],[1,0,1],[0,0,0]], np.uint8)) == 2) |
        (cv2.filter2D(skel, -1, np.array([[0,1,0],[0,0,0],[0,1,0]], np.uint8)) == 2) |
        (cv2.filter2D(skel, -1, np.array([[1,0,0],[0,0,0],[0,0,1]], np.uint8)) == 2) |
        (cv2.filter2D(skel, -1, np.array([[0,0,1],[0,0,0],[1,0,0]], np.uint8)) == 2)
    )
    bends   = (skel == 1) & (n == 2) & ~straight
    corners: list[tuple[int, int]] = []
    for mask in (endpoints, junctions, bends):
        ys, xs = np.where(mask)
        corners += [(int(x), int(y)) for x, y in zip(xs, ys)]
    return corners


def _suppress_neighbors(corners: list[tuple[int, int]], radius: int = 15) -> list[tuple[int, int]]:
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


# ── Graph refinement helpers ──────────────────────────────────────────────────

def _straighten_graph(G: nx.Graph, tol: int = 20) -> nx.Graph:
    """Snap near-collinear nodes to a shared coordinate (removes staircase artifacts)."""
    def cluster_snap(vals: np.ndarray, tol: int) -> np.ndarray:
        order  = np.argsort(vals)
        result = vals.copy()
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and vals[order[j]] - vals[order[i]] <= tol:
                j += 1
            result[order[i:j]] = round(vals[order[i:j]].mean())
            i = j
        return result
    nodes = list(G.nodes())
    xs    = np.array([G.nodes[n]["x"] for n in nodes], dtype=float)
    ys    = np.array([G.nodes[n]["y"] for n in nodes], dtype=float)
    xs_s, ys_s = cluster_snap(xs, tol), cluster_snap(ys, tol)
    for i, n in enumerate(nodes):
        G.nodes[n]["x"] = int(xs_s[i])
        G.nodes[n]["y"] = int(ys_s[i])
    return G


def _filter_small_components(G: nx.Graph, min_edges: int = 3) -> nx.Graph:
    """Drop connected components with fewer than min_edges (noise/dots)."""
    keep: set[int] = set()
    for comp in nx.connected_components(G):
        if G.subgraph(comp).number_of_edges() >= min_edges:
            keep.update(comp)
    return G.subgraph(keep).copy()


def _build_dag(G: nx.Graph) -> nx.DiGraph:
    """Convert undirected wall graph to a DAG via DFS spanning forest."""
    dag = nx.DiGraph()
    for n, attr in G.nodes(data=True):
        dag.add_node(n, **attr)
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        for parent, child in nx.dfs_edges(sub, source=min(comp)):
            dag.add_edge(parent, child, **sub[parent][child])
    return dag


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
    Find the two wall-graph corners that bound this opening gap.

    Strategy: collect all corner nodes within SNAP_PX of the expanded bbox,
    prioritising degree-1 nodes (wall terminations at the gap) over junctions.
    Then pick the closest such node, then the closest second node that lies on
    the *opposite side* of the opening centre — ensuring the pair spans the gap.
    """
    cx, cy   = (x1 + x2) / 2, (y1 + y2) / 2
    # expanded search box
    bx1, by1 = x1 - SNAP_PX, y1 - SNAP_PX
    bx2, by2 = x2 + SNAP_PX, y2 + SNAP_PX

    candidates: list[tuple[int, float, int]] = []   # (degree, dist², node_id)
    for n, attr in G.nodes(data=True):
        if attr["kind"] != "corner":
            continue
        nx_, ny_ = attr["x"], attr["y"]
        if not (bx1 <= nx_ <= bx2 and by1 <= ny_ <= by2):
            continue
        d2 = (nx_ - cx) ** 2 + (ny_ - cy) ** 2
        candidates.append((G.degree(n), d2, n))

    if not candidates:
        return []

    # degree-1 endpoints first, then by distance
    candidates.sort()

    selected: list[int] = []
    for _, _, nid in candidates:
        if not selected:
            selected.append(nid)
            continue
        # second node must be on the opposite side of the centre from the first
        n0   = G.nodes[selected[0]]
        nk   = G.nodes[nid]
        dot  = (n0["x"] - cx) * (nk["x"] - cx) + (n0["y"] - cy) * (nk["y"] - cy)
        if dot <= 0:
            selected.append(nid)
            break

    return selected


def attach_openings(G: nx.Graph, openings: list[dict[str, Any]]) -> nx.Graph:
    """
    Add each opening as a new node and connect it to the wall-gap corners.

    When two or more gap corners are found, a single direct edge is drawn
    between the first pair of corners (no routing through the center node).
    The center node is kept for JSON metadata but carries no graph edges in
    this case.  Falls back to center-connected edges when fewer than 2
    gap corners are found.
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

        if len(gap) >= 2:
            # Direct corner-to-corner edge; opening_node links back to metadata
            a, b = gap[0], gap[1]
            dist = ((G.nodes[a]["x"] - G.nodes[b]["x"]) ** 2 +
                    (G.nodes[a]["y"] - G.nodes[b]["y"]) ** 2) ** 0.5
            G.add_edge(a, b, length=dist, kind="opening", opening_node=next_id)
        else:
            for nid in gap:
                attr = G.nodes[nid]
                dist = ((attr["x"] - cx) ** 2 + (attr["y"] - cy) ** 2) ** 0.5
                G.add_edge(next_id, nid, length=dist, kind="opening")

        next_id += 1
    return G


# ── JSON export ───────────────────────────────────────────────────────────────

def _opening_offset_and_width(
    bbox: tuple,
    wx1: float, wy1: float,
    wx2: float, wy2: float,
) -> tuple[float, float]:
    """
    Project an opening bbox onto a wall line segment.

    Returns (offset, width) where:
      offset = distance from wall start to the near edge of the opening
      width  = length of the opening along the wall direction
    Both values are clamped to [0, wall_length].
    """
    x1, y1, x2, y2 = bbox
    dx, dy   = wx2 - wx1, wy2 - wy1
    wall_len = (dx ** 2 + dy ** 2) ** 0.5
    if wall_len == 0:
        return 0.0, 0.0
    ux, uy = dx / wall_len, dy / wall_len
    # project all four bbox corners onto the wall direction
    ts = [
        (cx - wx1) * ux + (cy - wy1) * uy
        for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    ]
    t_min = max(0.0, min(ts))
    t_max = min(wall_len, max(ts))
    return round(t_min, 1), round(max(0.0, t_max - t_min), 1)


def _nearest_wall_id(
    cx: float, cy: float,
    wall_records: list[dict],
) -> str | None:
    """Return the id of the wall whose line is closest to (cx, cy)."""
    best_id, best_d = None, float("inf")
    for w in wall_records:
        wx1, wy1, wx2, wy2 = w["_x1"], w["_y1"], w["_x2"], w["_y2"]
        dx, dy = wx2 - wx1, wy2 - wy1
        lsq = dx * dx + dy * dy
        if lsq == 0:
            continue
        t   = max(0.0, min(1.0, ((cx - wx1) * dx + (cy - wy1) * dy) / lsq))
        px  = wx1 + t * dx
        py  = wy1 + t * dy
        d   = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if d < best_d:
            best_d, best_id = d, w["id"]
    return best_id


def graph_to_json(G: nx.Graph) -> dict[str, Any]:
    """
    Convert the combined graph to the structured output JSON.

    Schema
    ------
    {
      "walls": [
        {"id": "wall_N", "start": {"x": px, "y": px}, "end": {"x": px, "y": px}},
        ...
      ],
      "openings": [
        {"id": "opening_N", "wallId": "wall_N",
         "offset": px, "width": px, "type": "door"|"window"},
        ...
      ]
    }

    Wall-merging: collinear wall segments on either side of an opening are
    merged into one span (≤30° tolerance).  Each opening is then referenced
    by wallId + offset so consumers can place it precisely on the wall.
    """
    # ── collect direct opening edges (corner-to-corner) ───────────────────────
    opening_edges: list[tuple[int, int, int]] = [
        (a, b, edata["opening_node"])
        for a, b, edata in G.edges(data=True)
        if edata.get("kind") == "opening" and "opening_node" in edata
    ]

    # ── wall merging ──────────────────────────────────────────────────────────
    absorbed: set[frozenset] = set()
    wall_records: list[dict] = []   # internal dicts with _x1/_y1/_x2/_y2 for math
    node_to_wall: dict[int, str] = {}  # opening_node_id → wall_id
    wall_idx = 0

    for ca, cb, nid in opening_edges:
        def _wall_nbrs(node: int) -> list[int]:
            return [
                nb for nb in G.neighbors(node)
                if G.nodes[nb]["kind"] == "corner"
                and G[node][nb].get("kind") == "wall"
            ]

        nbrs_a = _wall_nbrs(ca)
        nbrs_b = _wall_nbrs(cb)
        if not nbrs_a or not nbrs_b:
            continue

        dx_gap  = G.nodes[cb]["x"] - G.nodes[ca]["x"]
        dy_gap  = G.nodes[cb]["y"] - G.nodes[ca]["y"]
        gap_len = (dx_gap ** 2 + dy_gap ** 2) ** 0.5 or 1.0

        def _best(node: int, nbrs: list[int]) -> int | None:
            best_n, best_s = None, -1.0
            for nb in nbrs:
                dx = G.nodes[node]["x"] - G.nodes[nb]["x"]
                dy = G.nodes[node]["y"] - G.nodes[nb]["y"]
                seg_len = (dx ** 2 + dy ** 2) ** 0.5 or 1.0
                score   = abs((dx / seg_len) * (dx_gap / gap_len) +
                              (dy / seg_len) * (dy_gap / gap_len))
                if score > best_s:
                    best_s, best_n = score, nb
            return best_n if best_s > 0.85 else None

        p = _best(ca, nbrs_a)
        q = _best(cb, nbrs_b)
        if p is None or q is None:
            continue

        absorbed.add(frozenset({ca, p}))
        absorbed.add(frozenset({cb, q}))

        wall_idx += 1
        wid = f"wall_{wall_idx}"
        px, py = G.nodes[p]["x"], G.nodes[p]["y"]
        qx, qy = G.nodes[q]["x"], G.nodes[q]["y"]
        wall_records.append({
            "id":    wid,
            "start": {"x": px, "y": py},
            "end":   {"x": qx, "y": qy},
            "_x1": px, "_y1": py, "_x2": qx, "_y2": qy,
        })
        node_to_wall[nid] = wid

    # ── remaining (non-absorbed) wall edges ───────────────────────────────────
    for a, b, edata in G.edges(data=True):
        if edata.get("kind") == "wall" and frozenset({a, b}) not in absorbed:
            wall_idx += 1
            wid = f"wall_{wall_idx}"
            ax, ay = G.nodes[a]["x"], G.nodes[a]["y"]
            bx, by = G.nodes[b]["x"], G.nodes[b]["y"]
            wall_records.append({
                "id":    wid,
                "start": {"x": ax, "y": ay},
                "end":   {"x": bx, "y": by},
                "_x1": ax, "_y1": ay, "_x2": bx, "_y2": by,
            })

    # ── openings ──────────────────────────────────────────────────────────────
    wall_by_id = {w["id"]: w for w in wall_records}
    openings: list[dict] = []
    opening_idx = 0

    for n, attr in G.nodes(data=True):
        if attr["kind"] == "corner":
            continue

        cx, cy      = attr["x"], attr["y"]
        x1, y1, x2, y2 = attr["bbox"]

        # prefer the wall this opening was merged into; fall back to nearest
        wid = node_to_wall.get(n) or _nearest_wall_id(cx, cy, wall_records)
        if wid is None:
            continue

        w = wall_by_id[wid]
        offset, width = _opening_offset_and_width(
            (x1, y1, x2, y2),
            w["_x1"], w["_y1"], w["_x2"], w["_y2"],
        )

        opening_idx += 1
        kind = "door" if attr["kind"] in ("door", "2door") else "window"
        openings.append({
            "id":     f"opening_{opening_idx}",
            "wallId": wid,
            "offset": offset,
            "width":  width,
            "type":   kind,
        })

    # strip internal geometry keys before returning
    walls = [{"id": w["id"], "start": w["start"], "end": w["end"]} for w in wall_records]
    return {"walls": walls, "openings": openings}


# ── Public API ────────────────────────────────────────────────────────────────

def build_skeleton_graph(
    preprocessed: np.ndarray,
    spur_iter: int = 20,
    corner_radius: int = 15,
    tol: int = 20,
    min_edges: int = 3,
) -> SkeletonGraph:
    """
    Build a wall graph from a binary (cropped, preprocessed) floor plan image.

    Pipeline
    --------
    skeletonize → spur prune → topology corners → NMS →
    BFS edges → graph → straighten → filter noise → DAG

    Parameters
    ----------
    preprocessed  : binary np.ndarray — output of FloorPlanPreprocessor.process()
    spur_iter     : spur-pruning iterations (removes dangling skeleton tips)
    corner_radius : NMS suppression radius in pixels
    tol           : coordinate-snapping tolerance for graph straightening (px)
    min_edges     : minimum edges a component must have to survive noise filtering

    Returns
    -------
    SkeletonGraph with skeleton_raw, skeleton_thick, corners, graph, and dag
    """
    skeleton_raw    = (skeletonize(img_as_bool(preprocessed > 0)) * 255).astype(np.uint8)
    skeleton_pruned = _prune_spurs(skeleton_raw, iterations=spur_iter)
    skeleton_thick  = cv2.dilate(skeleton_pruned, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    corners = _suppress_neighbors(_detect_corners_topology(skeleton_pruned), radius=corner_radius)
    edges   = _bfs_edges(corners, skeleton_pruned)

    G = nx.Graph()
    for cid, (x, y) in enumerate(corners):
        G.add_node(cid, x=x, y=y, kind="corner")
    for key, length in edges.items():
        a, b = tuple(key)
        G.add_edge(a, b, length=int(length), kind="wall")

    G   = _straighten_graph(G, tol=tol)
    G   = _filter_small_components(G, min_edges=min_edges)
    dag = _build_dag(G)

    corners = [(attr["x"], attr["y"]) for _, attr in G.nodes(data=True)]

    return SkeletonGraph(
        skeleton_raw=skeleton_pruned,
        skeleton_thick=skeleton_thick,
        corners=corners,
        graph=G,
        dag=dag,
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

    # Supplement YOLO with template-matching detection for windows and doors.
    # YOLO always wins: pass all YOLO bboxes so any FM hit that overlaps is dropped.
    gray = cv2.cvtColor(prep.raw_no_text, cv2.COLOR_BGR2GRAY)
    yolo_bboxes = [det["bbox"] for det in openings]
    fm_openings = detect_windows_fm(gray, existing_bboxes=yolo_bboxes)
    openings = openings + fm_openings   # YOLO first (higher conf / priority)

    # Work on a copy so skel.graph stays pure (corners + walls only)
    combined = skel.graph.copy()
    attach_openings(combined, openings)

    return FloorPlanOutput(
        prep=prep,
        skel=skel,
        graph=combined,
        json=graph_to_json(combined),
    )


# Existing notebook logic lives in this module.
# Keep detection, preprocessing, skeletonization, graph fusion, and JSON export here.
# Only add thin wrappers in app.py or other API modules.
def image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR OpenCV image."""
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image bytes: unable to decode image.")
    return image


def process_floorplan_bytes(
    image_bytes: bytes,
    model_path: str | Path = "Models/best_v2.pt",
    **kwargs,
) -> dict[str, Any]:
    """Process raw image bytes and return the exact pipeline JSON output."""
    image = image_bytes_to_bgr(image_bytes)
    return run_pipeline(image, model_path=model_path, **kwargs).json


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    img_path   = sys.argv[1] if len(sys.argv) > 1 else "data/PNG/raw/IIa_A01001.png"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "Models/best_v2.pt"

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(img_path)

    out = run_pipeline(image, model_path=model_path)
    G   = out.graph

    n_doors   = sum(1 for o in out.json["openings"] if o["type"] == "door")
    n_windows = sum(1 for o in out.json["openings"] if o["type"] == "window")
    print(f"rotation_angle : {out.prep.rotation_angle}°")
    print(f"crop_bbox      : {out.prep.crop_bbox}")
    print(f"corners        : {len(out.skel.corners)}")
    print(f"walls          : {len(out.json['walls'])}")
    print(f"doors          : {n_doors}")
    print(f"windows        : {n_windows}")
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
