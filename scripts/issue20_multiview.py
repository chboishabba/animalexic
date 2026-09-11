"""Known-pose multi-camera adapter for Pixeltovoxelprojector issue #20 data.

This module deliberately stops at the producer boundary: it validates the contributed
camera/frame metadata, maps image evidence to world rays using the source convention,
and accumulates sparse motion evidence into candidate voxels. Canonical promotion is
owned by Animalexic's existing governance layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


class MetadataError(ValueError):
    """Raised when known-pose metadata is incomplete or ambiguous."""


@dataclass(frozen=True)
class CameraObservation:
    camera_id: int
    frame_index: int
    position: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    fov_deg: float
    image_file: str
    pose_source: str = "issue20_known_pose"


def _number(row: Mapping[str, object], key: str) -> float:
    if key not in row or isinstance(row[key], bool):
        raise MetadataError(f"missing numeric field: {key}")
    try:
        value = float(row[key])
    except (TypeError, ValueError) as exc:
        raise MetadataError(f"invalid numeric field: {key}") from exc
    if not math.isfinite(value):
        raise MetadataError(f"non-finite numeric field: {key}")
    return value


def _integer(row: Mapping[str, object], key: str) -> int:
    value = _number(row, key)
    if not value.is_integer():
        raise MetadataError(f"field {key} must be integral")
    return int(value)


def load_issue20_metadata(path: str | Path) -> list[CameraObservation]:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetadataError(f"cannot read metadata: {path}") from exc
    if not isinstance(raw, list) or not raw:
        raise MetadataError("metadata top level must be a non-empty array")

    out: list[CameraObservation] = []
    seen: set[tuple[int, int]] = set()
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise MetadataError(f"entry {i} is not an object")
        camera_id = _integer(row, "camera_index")
        frame_index = _integer(row, "frame_index")
        if (camera_id, frame_index) in seen:
            raise MetadataError(f"duplicate camera/frame pair: {camera_id}/{frame_index}")
        seen.add((camera_id, frame_index))

        pos = row.get("camera_position")
        if not isinstance(pos, list) or len(pos) != 3:
            raise MetadataError(f"entry {i} requires camera_position[3]")
        try:
            position = tuple(float(x) for x in pos)
        except (TypeError, ValueError) as exc:
            raise MetadataError(f"entry {i} has invalid camera_position") from exc
        if not all(math.isfinite(x) for x in position):
            raise MetadataError(f"entry {i} has non-finite camera_position")

        image_file = row.get("image_file")
        if not isinstance(image_file, str) or not image_file.strip():
            raise MetadataError(f"entry {i} requires image_file")

        fov = _number(row, "fov_degrees")
        if not (0.0 < fov < 180.0):
            raise MetadataError(f"entry {i} has invalid fov_degrees")

        out.append(
            CameraObservation(
                camera_id=camera_id,
                frame_index=frame_index,
                position=(position[0], position[1], position[2]),
                yaw_deg=_number(row, "yaw"),
                pitch_deg=_number(row, "pitch"),
                roll_deg=_number(row, "roll"),
                fov_deg=fov,
                image_file=image_file,
            )
        )
    out.sort(key=lambda o: (o.frame_index, o.camera_id))
    return out


def group_synchronised_frames(
    observations: Iterable[CameraObservation],
    required_camera_ids: set[int] | None = None,
) -> dict[int, list[CameraObservation]]:
    by_frame: dict[int, list[CameraObservation]] = defaultdict(list)
    for obs in observations:
        by_frame[obs.frame_index].append(obs)
    if required_camera_ids is None:
        required_camera_ids = {o.camera_id for group in by_frame.values() for o in group}
    result: dict[int, list[CameraObservation]] = {}
    for frame_index in sorted(by_frame):
        group = sorted(by_frame[frame_index], key=lambda o: o.camera_id)
        if required_camera_ids.issubset({o.camera_id for o in group}):
            result[frame_index] = group
    return result


def _matmul(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        sum(a[row * 3 + k] * b[k * 3 + col] for k in range(3))
        for row in range(3)
        for col in range(3)
    )


def _rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> tuple[float, ...]:
    # Match the referenced Rust producer exactly: yaw Rz, roll Ry, pitch Rx.
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    rz = (cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0)
    ry = (cr, 0.0, sr, 0.0, 1.0, 0.0, -sr, 0.0, cr)
    rx = (1.0, 0.0, 0.0, 0.0, cp, -sp, 0.0, sp, cp)
    return _matmul(_matmul(rz, ry), rx)


def _normalize(v: Sequence[float]) -> tuple[float, float, float]:
    n = math.sqrt(sum(x * x for x in v))
    if n <= 1e-12:
        raise ValueError("zero-length ray direction")
    return (v[0] / n, v[1] / n, v[2] / n)


def pixel_ray_world(
    camera: CameraObservation,
    *,
    u: float,
    v: float,
    width: int,
    height: int,
) -> tuple[float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    focal = (width / 2.0) / math.tan(math.radians(camera.fov_deg) / 2.0)
    local = _normalize((u - width / 2.0, -(v - height / 2.0), focal))
    m = _rotation_matrix(camera.yaw_deg, camera.pitch_deg, camera.roll_deg)
    world = (
        m[0] * local[0] + m[1] * local[1] + m[2] * local[2],
        m[3] * local[0] + m[4] * local[1] + m[5] * local[2],
        m[6] * local[0] + m[7] * local[1] + m[8] * local[2],
    )
    return _normalize(world)


def accumulate_sparse_motion_voxels(
    *,
    cameras: Mapping[int, CameraObservation],
    samples: Iterable[Mapping[str, object]],
    voxel_size: float,
    max_distance: float,
    min_camera_support: int = 2,
    step_fraction: float = 0.25,
) -> dict[tuple[int, int, int], dict[str, object]]:
    if voxel_size <= 0 or max_distance <= 0:
        raise ValueError("voxel_size and max_distance must be positive")
    if min_camera_support < 1:
        raise ValueError("min_camera_support must be at least one")
    step = voxel_size * step_fraction
    if step <= 0:
        raise ValueError("step_fraction must be positive")

    evidence: dict[tuple[int, int, int], float] = defaultdict(float)
    supporters: dict[tuple[int, int, int], set[int]] = defaultdict(set)

    for sample in samples:
        camera_id = int(sample["camera_id"])
        if camera_id not in cameras:
            raise MetadataError(f"motion sample references unknown camera {camera_id}")
        camera = cameras[camera_id]
        if "direction" in sample:
            direction = _normalize(tuple(float(x) for x in sample["direction"]))
        else:
            for key in ("u", "v", "width", "height"):
                if key not in sample:
                    raise MetadataError(f"motion sample missing {key}")
            direction = pixel_ray_world(
                camera,
                u=float(sample["u"]),
                v=float(sample["v"]),
                width=int(sample["width"]),
                height=int(sample["height"]),
            )
        weight = float(sample.get("evidence", 1.0))
        if not math.isfinite(weight) or weight <= 0:
            continue

        visited: set[tuple[int, int, int]] = set()
        distance = 0.0
        while distance <= max_distance:
            point = tuple(camera.position[i] + direction[i] * distance for i in range(3))
            key = tuple(math.floor(coord / voxel_size + 0.5) for coord in point)
            if key not in visited:
                evidence[key] += weight
                supporters[key].add(camera_id)
                visited.add(key)
            distance += step

    result: dict[tuple[int, int, int], dict[str, object]] = {}
    for key, total in evidence.items():
        support = supporters[key]
        if len(support) >= min_camera_support:
            result[key] = {
                "evidence": total,
                "camera_support": len(support),
                "camera_ids": tuple(sorted(support)),
                "status": "candidate",
                "producer": "issue20_known_pose_sparse_motion",
            }
    return result
