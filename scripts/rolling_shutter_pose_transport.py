from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RollingShutterReadoutCandidate:
    readout_time_s: float
    direction: str
    source_reference: str
    status: str = "candidate"
    readout_calibration_paid: bool = False


@dataclass(frozen=True)
class RowPoseCandidate:
    capture_time_s: float
    position_world_m: tuple[float, float, float]
    rotation_world_from_camera: tuple[float, ...]
    readout_source_reference: str
    status: str = "candidate"
    readout_calibration_paid: bool = False


def _validate_readout(readout: RollingShutterReadoutCandidate) -> None:
    if readout.status != "candidate":
        raise ValueError("rolling-shutter readout must be a candidate")
    if not math.isfinite(readout.readout_time_s) or readout.readout_time_s < 0:
        raise ValueError("readout_time_s must be finite and non-negative")
    if readout.direction not in {"top_to_bottom", "bottom_to_top"}:
        raise ValueError("unsupported rolling-shutter direction")
    if not readout.source_reference:
        raise ValueError("readout source reference is required")


def row_capture_time_offset_s(
    row: int,
    image_height: int,
    readout: RollingShutterReadoutCandidate,
) -> float:
    """Return row time relative to the nominal frame-centre timestamp."""
    _validate_readout(readout)
    if image_height < 2:
        raise ValueError("image_height must be at least two")
    if not 0 <= row < image_height:
        raise ValueError("row is outside image")
    fraction = float(row) / float(image_height - 1)
    if readout.direction == "bottom_to_top":
        fraction = 1.0 - fraction
    return (fraction - 0.5) * float(readout.readout_time_s)


def _project_so3(matrix: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(matrix)
    correction = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        correction[-1, -1] = -1.0
    return U @ correction @ Vt


def _rotation_power(rotation: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate an SO(3) rotation by axis-angle exponentiation."""
    R = _project_so3(rotation)
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return np.eye(3)
    if abs(math.pi - theta) < 1e-7:
        # Stable axis extraction near pi from the symmetric part.
        values, vectors = np.linalg.eigh((R + np.eye(3)) / 2.0)
        axis = vectors[:, int(np.argmax(values))]
        axis = axis / (np.linalg.norm(axis) + 1e-15)
    else:
        axis = np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=np.float64,
        ) / (2.0 * math.sin(theta))
    angle = theta * float(fraction)
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def interpolate_row_pose(
    *,
    frame_time_s: float,
    row: int,
    image_height: int,
    readout: RollingShutterReadoutCandidate,
    before,
    after,
) -> RowPoseCandidate:
    """Interpolate a candidate camera pose at the row capture time.

    This transports an already-supplied readout model through a candidate
    trajectory. It does not estimate readout time/direction or pay calibration.
    """
    offset = row_capture_time_offset_s(row, image_height, readout)
    capture_time = float(frame_time_s) + offset
    if getattr(before, "status", None) != "candidate" or getattr(after, "status", None) != "candidate":
        raise ValueError("rolling-shutter interpolation requires candidate keyframes")
    t0 = float(before.time_s)
    t1 = float(after.time_s)
    if not math.isfinite(t0) or not math.isfinite(t1) or t1 <= t0:
        raise ValueError("keyframe times must be finite and ordered")
    if capture_time < t0 - 1e-12 or capture_time > t1 + 1e-12:
        raise ValueError("row capture time is outside bracketing keyframes")
    fraction = (capture_time - t0) / (t1 - t0)

    p0 = np.asarray(before.position_world_m, dtype=np.float64)
    p1 = np.asarray(after.position_world_m, dtype=np.float64)
    R0 = np.asarray(before.rotation_world_from_camera, dtype=np.float64).reshape(3, 3)
    R1 = np.asarray(after.rotation_world_from_camera, dtype=np.float64).reshape(3, 3)
    if p0.shape != (3,) or p1.shape != (3,) or not all(np.all(np.isfinite(x)) for x in (p0, p1, R0, R1)):
        raise ValueError("keyframe poses must be finite")

    position = (1.0 - fraction) * p0 + fraction * p1
    relative = R0.T @ R1
    rotation = R0 @ _rotation_power(relative, fraction)
    return RowPoseCandidate(
        capture_time_s=capture_time,
        position_world_m=tuple(float(x) for x in position),
        rotation_world_from_camera=tuple(float(x) for x in rotation.reshape(-1)),
        readout_source_reference=readout.source_reference,
        status="candidate",
        readout_calibration_paid=False,
    )
