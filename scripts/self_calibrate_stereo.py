#!/usr/bin/env python3

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np


def _load_pair(left_path: Path | None, right_path: Path | None, sbs_path: Path | None):
    import cv2

    if sbs_path is not None:
        frame = cv2.imread(str(sbs_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError(f"failed to read SBS image: {sbs_path}")
        mid = frame.shape[1] // 2
        return frame[:, :mid], frame[:, mid:]

    if left_path is None or right_path is None:
        raise RuntimeError("provide either --sbs-image or both --left-image/--right-image")

    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise RuntimeError("failed to read one or both input images")
    return left, right


def _homography_to_maps(H: np.ndarray, width: int, height: int):
    import cv2

    H_inv = np.linalg.inv(H)
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    ones = np.ones_like(xs)
    pts = np.stack([xs, ys, ones], axis=-1).reshape(-1, 3).T
    warped = H_inv @ pts
    warped /= np.clip(warped[2:3], 1e-6, None)
    map_x = warped[0].reshape(height, width).astype(np.float32)
    map_y = warped[1].reshape(height, width).astype(np.float32)
    return map_x, map_y


def _pseudo_q(width: int, height: int):
    f = float(max(width, height))
    cx = width * 0.5
    cy = height * 0.5
    baseline = 1.0
    return np.array(
        [
            [1.0, 0.0, 0.0, -cx],
            [0.0, 1.0, 0.0, -cy],
            [0.0, 0.0, 0.0, f],
            [0.0, 0.0, -1.0 / baseline, 0.0],
        ],
        dtype=np.float32,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-image", type=str, help="Left image path")
    ap.add_argument("--right-image", type=str, help="Right image path")
    ap.add_argument("--sbs-image", type=str, help="Side-by-side stereo image path")
    ap.add_argument("--rig-id", type=str, default="rig_selfcal", help="Logical rig identifier")
    ap.add_argument("--max-features", type=int, default=4000, help="ORB feature budget")
    ap.add_argument("--output", required=True, help="Output .npz path for self-calibration artifact")
    ap.add_argument("--save-debug-dir", type=str, help="Optional directory for rectified debug images")
    args = ap.parse_args()

    import cv2

    left, right = _load_pair(
        Path(args.left_image) if args.left_image else None,
        Path(args.right_image) if args.right_image else None,
        Path(args.sbs_image) if args.sbs_image else None,
    )
    if left.shape != right.shape:
        raise RuntimeError(f"left/right sizes differ: {left.shape} vs {right.shape}")

    height, width = left.shape
    orb = cv2.ORB_create(nfeatures=args.max_features)
    kp_l, des_l = orb.detectAndCompute(left, None)
    kp_r, des_r = orb.detectAndCompute(right, None)
    if des_l is None or des_r is None:
        raise RuntimeError("ORB failed to find enough features")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des_l, des_r, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 16:
        raise RuntimeError(f"not enough feature matches for self-calibration: {len(good)}")

    pts_l = np.float32([kp_l[m.queryIdx].pt for m in good])
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in good])
    F, mask = cv2.findFundamentalMat(pts_l, pts_r, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None or mask is None:
        raise RuntimeError("failed to estimate fundamental matrix")
    inliers = mask.ravel().astype(bool)
    pts_l_in = pts_l[inliers]
    pts_r_in = pts_r[inliers]
    if len(pts_l_in) < 12:
        raise RuntimeError(f"not enough inliers after RANSAC: {len(pts_l_in)}")

    ok, H1, H2 = cv2.stereoRectifyUncalibrated(pts_l_in, pts_r_in, F, imgSize=(width, height))
    if not ok:
        raise RuntimeError("stereoRectifyUncalibrated failed")

    left_map_x, left_map_y = _homography_to_maps(H1, width, height)
    right_map_x, right_map_y = _homography_to_maps(H2, width, height)
    q_matrix = _pseudo_q(width, height)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_id = str(uuid.uuid4())
    np.savez_compressed(
        output_path,
        left_map_x=left_map_x,
        left_map_y=left_map_y,
        right_map_x=right_map_x,
        right_map_y=right_map_y,
        H1=H1,
        H2=H2,
        F=F,
        Q=q_matrix,
    )

    metadata = {
        "schema_version": "fixed_rig_selfcal_v1",
        "calibration_id": calibration_id,
        "rig_id": args.rig_id,
        "image_width": width,
        "image_height": height,
        "image_size": {"width": width, "height": height},
        "method": "orb_fundamental_uncalibrated_rectification",
        "quality": {
            "feature_count_left": len(kp_l),
            "feature_count_right": len(kp_r),
            "match_count": len(good),
            "inlier_count": int(inliers.sum()),
            "inlier_ratio": float(inliers.mean()),
        },
        "provenance": {
            "left_image": args.left_image,
            "right_image": args.right_image,
            "sbs_image": args.sbs_image,
            "created_at_unix": time.time(),
        },
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    if args.save_debug_dir:
        debug_dir = Path(args.save_debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        rect_left = cv2.remap(left, left_map_x, left_map_y, interpolation=cv2.INTER_LINEAR)
        rect_right = cv2.remap(right, right_map_x, right_map_y, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(debug_dir / "rectified_left.png"), rect_left)
        cv2.imwrite(str(debug_dir / "rectified_right.png"), rect_right)

    print(f"wrote self-calibration artifact: {output_path}")
    print(f"wrote self-calibration metadata: {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
