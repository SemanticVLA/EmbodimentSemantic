from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..proxy.bbox_geometry import bbox_iou
from ..proxy.schemas import BBoxObject


def _features(cv2: Any, gray: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    x1, y1, x2, y2 = np.round(bbox).astype(int)
    height, width = gray.shape
    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(y1 + 1, min(y2, height))
    mask = np.zeros_like(gray)
    mask[y1:y2, x1:x2] = 255
    return cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        maxCorners=120,
        qualityLevel=0.005,
        minDistance=3,
        blockSize=5,
    )


def _track_bbox_series(
    cv2: Any,
    images: list[np.ndarray],
    initial_bbox: tuple[float, ...],
) -> list[np.ndarray]:
    if not images:
        return []
    bbox = np.asarray(initial_bbox, dtype=np.float32)
    output = [bbox.copy()]
    gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    points = _features(cv2, gray, bbox)
    for image in images[1:]:
        next_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if points is None or len(points) < 4:
            break
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            next_gray,
            points,
            None,
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.02),
        )
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            next_gray,
            gray,
            moved,
            None,
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.02),
        )
        valid = (
            (status.ravel() == 1)
            & (backward_status.ravel() == 1)
            & (
                np.linalg.norm(
                    points.reshape(-1, 2) - backward.reshape(-1, 2), axis=1
                )
                < 2.0
            )
        )
        old = points.reshape(-1, 2)[valid]
        new = moved.reshape(-1, 2)[valid]
        if len(old) < 4:
            break
        transform, _ = cv2.estimateAffinePartial2D(
            old,
            new,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=2000,
            confidence=0.99,
        )
        if transform is None:
            break
        scale = float(np.hypot(transform[0, 0], transform[0, 1]))
        if not 0.92 <= scale <= 1.08:
            break
        corners = np.asarray(
            [[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32
        ).reshape(-1, 1, 2)
        transformed = cv2.transform(corners, transform).reshape(-1, 2)
        xs = sorted(transformed[:, 0])
        ys = sorted(transformed[:, 1])
        bbox = np.asarray([xs[0], ys[0], xs[1], ys[1]], dtype=np.float32)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, image.shape[1])
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, image.shape[0])
        if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
            break
        gray = next_gray
        points = _features(cv2, gray, bbox)
        output.append(bbox.copy())
    return output


def _track_bbox(cv2: Any, images: list[np.ndarray], initial_bbox: tuple[float, ...]) -> np.ndarray | None:
    series = _track_bbox_series(cv2, images, initial_bbox)
    return series[-1] if len(series) == len(images) else None


def _fuse_flow_boxes(
    forward: np.ndarray | None,
    backward: np.ndarray | None,
    *,
    minimum_iou: float,
) -> np.ndarray | None:
    if forward is not None and backward is not None:
        if bbox_iou(tuple(forward), tuple(backward)) < minimum_iou:
            return None
        return (forward + backward) / 2.0
    return forward if forward is not None else backward


def track_missing_bowl(
    *,
    video_path: str | Path,
    video_start_timestamp: float,
    fps: float,
    previous_frame: int,
    target_frame: int,
    next_frame: int,
    previous: BBoxObject,
    following: BBoxObject,
    minimum_bidirectional_iou: float = 0.15,
) -> BBoxObject | None:
    return track_bowl_sequence(
        video_path=video_path,
        video_start_timestamp=video_start_timestamp,
        fps=fps,
        previous_frame=previous_frame,
        target_frames=[target_frame],
        next_frame=next_frame,
        previous=previous,
        following=following,
        minimum_bidirectional_iou=minimum_bidirectional_iou,
    ).get(target_frame)


def track_bowl_sequence(
    *,
    video_path: str | Path,
    video_start_timestamp: float,
    fps: float,
    previous_frame: int,
    target_frames: list[int],
    next_frame: int,
    previous: BBoxObject,
    following: BBoxObject,
    minimum_bidirectional_iou: float = 0.15,
) -> dict[int, BBoxObject]:
    requested = sorted(
        set(frame for frame in target_frames if previous_frame < frame < next_frame)
    )
    if not requested:
        return {}
    try:
        import cv2
    except ImportError:
        return {}

    global_start = int(round(video_start_timestamp * fps)) + previous_frame
    capture = cv2.VideoCapture(str(Path(video_path)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, global_start)
    images: list[np.ndarray] = []
    try:
        for _ in range(previous_frame, next_frame + 1):
            ok, image = capture.read()
            if not ok:
                return {}
            images.append(image)
    finally:
        capture.release()

    forward = _track_bbox_series(cv2, images, previous.bbox)
    backward = _track_bbox_series(cv2, list(reversed(images)), following.bbox)
    output: dict[int, BBoxObject] = {}
    for target_frame in requested:
        forward_offset = target_frame - previous_frame
        backward_offset = next_frame - target_frame
        forward_box = forward[forward_offset] if forward_offset < len(forward) else None
        backward_box = backward[backward_offset] if backward_offset < len(backward) else None
        fused = _fuse_flow_boxes(
            forward_box,
            backward_box,
            minimum_iou=minimum_bidirectional_iou,
        )
        if fused is None:
            continue
        output[target_frame] = BBoxObject(
            bbox=tuple(float(value) for value in fused),
            confidence=0.70 * min(previous.confidence, following.confidence),
            tracking_confidence=0.60
            * min(previous.tracking_confidence, following.tracking_confidence),
            visible=previous.visible and following.visible,
            source="agent_bowl_dynamic_optical_flow",
        )
    return output
