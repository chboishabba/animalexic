from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
from scripts.camera_pose_fibre import recover_relative_pose_from_correspondences


@dataclass(frozen=True)
class KnownPoseScore:
    rotation_error_deg: float
    translation_direction_error_deg: float
    inlier_count: int
    inlier_ratio: float


def derive_camera_matrix_from_fov(width: int, height: int, fov_degrees: float):
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not (0.0 < fov_degrees < 180.0):
        raise ValueError("fov_degrees must lie between zero and 180 degrees")
    f = (width / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _static_feature_mask(image, dynamic_mask):
    if dynamic_mask is None:
        return None
    mask = np.asarray(dynamic_mask)
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError("dynamic mask shape must match image")
    return np.where(mask > 0, 0, 255).astype(np.uint8)


def match_static_features(
    image1,
    image2,
    *,
    dynamic_mask1=None,
    dynamic_mask2=None,
    min_matches: int = 8,
    max_features: int = 3000,
    ratio_test: float = 0.72,
):
    """Match only pose-admissible static-scene image features."""
    import cv2

    a = np.asarray(image1)
    b = np.asarray(image2)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("feature matching requires grayscale images")

    detector = cv2.SIFT_create(nfeatures=max_features)
    k1, d1 = detector.detectAndCompute(a, _static_feature_mask(a, dynamic_mask1))
    k2, d2 = detector.detectAndCompute(b, _static_feature_mask(b, dynamic_mask2))
    if d1 is None or d2 is None:
        raise ValueError("insufficient visual features")

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = []
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < ratio_test * pair[1].distance:
            good.append(pair[0])
    if len(good) < min_matches:
        raise ValueError(f"insufficient static feature matches: {len(good)} < {min_matches}")

    p1 = np.array([k1[m.queryIdx].pt for m in good], dtype=np.float64)
    p2 = np.array([k2[m.trainIdx].pt for m in good], dtype=np.float64)
    return p1, p2


def recover_pose_from_image_pair(
    image1_path,
    image2_path,
    camera_matrix,
    *,
    dynamic_mask1=None,
    dynamic_mask2=None,
    min_matches: int = 12,
    ransac_threshold_px: float = 1.5,
):
    """Recover a candidate relative pose from a pair of grayscale images."""
    import cv2

    a = cv2.imread(str(Path(image1_path)), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(Path(image2_path)), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        raise ValueError("cannot load image pair")
    p1, p2 = match_static_features(
        a,
        b,
        dynamic_mask1=dynamic_mask1,
        dynamic_mask2=dynamic_mask2,
        min_matches=min_matches,
    )
    return recover_relative_pose_from_correspondences(
        p1,
        p2,
        camera_matrix,
        ransac_threshold_px=ransac_threshold_px,
    )


def _angle_deg(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def score_recovered_pose_against_known(
    estimate,
    known_rotation_cam2_from_cam1,
    known_camera2_center_in_cam1,
):
    """Post-hoc oracle score; known pose is not an input to recovery."""
    R = np.asarray(estimate.rotation_cam2_from_cam1, dtype=np.float64).reshape(3, 3)
    Rk = np.asarray(known_rotation_cam2_from_cam1, dtype=np.float64).reshape(3, 3)
    dR = R @ Rk.T
    rot = math.degrees(
        math.acos(float(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
    )
    tdir = np.asarray(estimate.translation_direction_cam2_from_cam1, dtype=np.float64)
    center_dir = -R.T @ tdir
    trans = _angle_deg(center_dir, known_camera2_center_in_cam1)
    return KnownPoseScore(
        rot,
        trans,
        int(estimate.inlier_count),
        float(estimate.inlier_ratio),
    )


def _world_from_camera_rotation(observation):
    """Reproduce the Issue-20 yaw(Z), roll(Y), pitch(X) source convention."""
    yaw = math.radians(float(observation.yaw_deg))
    pitch = math.radians(float(observation.pitch_deg))
    roll = math.radians(float(observation.roll_deg))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    return rz @ ry @ rx


def known_relative_pose_from_observations(anchor, target):
    """Convert two known Issue-20 observations to validation geometry."""
    A1 = _world_from_camera_rotation(anchor)
    A2 = _world_from_camera_rotation(target)
    C1 = np.asarray(anchor.position, dtype=np.float64)
    C2w = np.asarray(target.position, dtype=np.float64)
    rotation_cam2_from_cam1 = A2.T @ A1
    camera2_center_in_cam1 = A1.T @ (C2w - C1)
    return rotation_cam2_from_cam1, camera2_center_in_cam1


def recover_and_score_issue20_pair(
    anchor,
    target,
    image_root,
    *,
    dynamic_mask1=None,
    dynamic_mask2=None,
    min_matches: int = 12,
    ransac_threshold_px: float = 1.5,
):
    """Recover from Issue-20 images, then score against metadata post hoc.

    The known camera poses are used only by the returned validation score. They
    are not used by the learned relative-pose producer and they do not pay
    metric scale for that producer.
    """
    import cv2

    root = Path(image_root)
    a = cv2.imread(str(root / anchor.image_file), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(root / target.image_file), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        raise ValueError("cannot load Issue-20 image pair")
    if a.shape != b.shape:
        raise ValueError("Issue-20 image pair dimensions must match")
    if abs(float(anchor.fov_deg) - float(target.fov_deg)) > 1e-9:
        raise ValueError("pair recovery currently requires matching camera FOV")

    h, w = a.shape[:2]
    K = derive_camera_matrix_from_fov(w, h, float(anchor.fov_deg))
    p1, p2 = match_static_features(
        a,
        b,
        dynamic_mask1=dynamic_mask1,
        dynamic_mask2=dynamic_mask2,
        min_matches=min_matches,
    )
    estimate = recover_relative_pose_from_correspondences(
        p1,
        p2,
        K,
        ransac_threshold_px=ransac_threshold_px,
    )
    known_R, known_C = known_relative_pose_from_observations(anchor, target)
    return estimate, score_recovered_pose_against_known(estimate, known_R, known_C)
