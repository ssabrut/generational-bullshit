from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from skimage import img_as_bool
from skimage.morphology import skeletonize

from preprocessing import FloorPlanPreprocessor, PipelineResult


@dataclass
class SkeletonGraph:
    skeleton_raw: np.ndarray        # 1-px binary skeleton
    skeleton_thick: np.ndarray      # dilated skeleton used for corner detection
    corners: list[tuple[int, int]]  # NMS-filtered corners snapped onto skeleton
    graph: nx.Graph                 # undirected wall graph (nodes=corners, edges=walls)


# ── Binary cleanup helpers ────────────────────────────────────────────────────

def _crop_to_content(binary: np.ndarray, margin: int = -10) -> np.ndarray:
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return binary
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    H, W = binary.shape
    y_min = max(0, y_min + margin)
    x_min = max(0, x_min + margin)
    y_max = min(H, y_max - margin)
    x_max = min(W, x_max - margin)
    return binary[y_min:y_max, x_min:x_max]


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
    binary = cv2.threshold(preprocessed, 127, 255, cv2.THRESH_BINARY)[1]
    cropped = _crop_to_content(binary, margin=-10)
    cleaned = _remove_noise(cropped, min_area_ratio=0.0001)
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
        G.add_node(cid, x=x, y=y)
    for key, length in edge_dict.items():
        a, b = tuple(key)
        G.add_edge(a, b, length=int(length))

    return SkeletonGraph(
        skeleton_raw=skeleton_raw,
        skeleton_thick=skeleton_thick,
        corners=snapped,
        graph=G,
    )


def run_pipeline(
    image: np.ndarray,
    preprocessor: FloorPlanPreprocessor | None = None,
) -> tuple[PipelineResult, SkeletonGraph]:
    """Run preprocessing (steps 1–3) then skeleton graph (step 4) on a BGR image."""
    if preprocessor is None:
        preprocessor = FloorPlanPreprocessor(angle_step=90)
    prep = preprocessor.process(image)
    skel = build_skeleton_graph(prep.preprocessed)
    return prep, skel


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/PNG/raw/IIa_A01001.png"
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(img_path)

    prep, skel = run_pipeline(image)

    G = skel.graph
    print(f"rotation_angle : {prep.rotation_angle}°")
    print(f"crop_bbox      : {prep.crop_bbox}")
    print(f"corners        : {len(skel.corners)}")
    print(f"graph          : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"components     : {nx.number_connected_components(G)}")

    out_dir = Path("data/PNG")
    stem = Path(img_path).stem

    (out_dir / "no_text").mkdir(parents=True, exist_ok=True)
    (out_dir / "crop_and_pad").mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "no_text" / f"{stem}.png"), prep.raw_no_text)
    cv2.imwrite(str(out_dir / "crop_and_pad" / f"{stem}.png"), prep.preprocessed)

    # Overlay graph on skeleton for visual inspection
    vis = cv2.cvtColor(skel.skeleton_raw, cv2.COLOR_GRAY2BGR)
    for a, b in G.edges():
        pa = (G.nodes[a]["x"], G.nodes[a]["y"])
        pb = (G.nodes[b]["x"], G.nodes[b]["y"])
        cv2.line(vis, pa, pb, (0, 255, 0), 2)
    for n, attr in G.nodes(data=True):
        p = (attr["x"], attr["y"])
        cv2.circle(vis, p, 6, (0, 0, 255), -1)
        cv2.circle(vis, p, 6, (255, 255, 255), 2)

    out_path = str(out_dir / f"{stem}_skeleton_graph.png")
    cv2.imwrite(out_path, vis)
    print(f"Saved graph overlay → {out_path}")
