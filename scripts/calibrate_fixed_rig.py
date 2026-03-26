#!/usr/bin/env python3

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np


def _read_image_pairs(left_dir: Path, right_dir: Path):
    import cv2

    left_files = sorted([p for p in left_dir.iterdir() if p.is_file()])
    right_files = {p.name: p for p in right_dir.iterdir() if p.is_file()}
    pairs = []
    for left_path in left_files:
        right_path = right_files.get(left_path.name)
        if right_path is None:
            continue
        left_img = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if left_img is None or right_img is None:
            continue
        pairs.append((left_path.name, left_img, right_img))
    if not pairs:
        raise RuntimeError("no matching readable calibration image pairs found")
    return pairs


def _epipolar_error(F: np.ndarray, points_l, points_r) -> tuple[float, float]:
    import cv2

    errs = []
    for pts_l, pts_r in zip(points_l, points_r):
        lines_l = cv2.computeCorrespondEpilines(pts_r.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
        lines_r = cv2.computeCorrespondEpilines(pts_l.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
        for p_l, p_r, line_l, line_r in zip(pts_l, pts_r, lines_l, lines_r):
            err_l = abs(line_l[0] * p_l[0] + line_l[1] * p_l[1] + line_l[2]) / max(1e-6, np.hypot(line_l[0], line_l[1]))
            err_r = abs(line_r[0] * p_r[0] + line_r[1] * p_r[1] + line_r[2]) / max(1e-6, np.hypot(line_r[0], line_r[1]))
            errs.append(0.5 * (float(err_l) + float(err_r)))
    if not errs:
        return 0.0, 0.0
    return float(np.mean(errs)), float(np.max(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-dir", required=True, help="Directory of left calibration images")
    ap.add_argument("--right-dir", required=True, help="Directory of right calibration images")
    ap.add_argument("--squares-x", type=int, required=True, help="ChArUco board square count in X")
    ap.add_argument("--squares-y", type=int, required=True, help="ChArUco board square count in Y")
    ap.add_argument("--square-length", type=float, required=True, help="Square size in user units")
    ap.add_argument("--marker-length", type=float, required=True, help="Marker size in user units")
    ap.add_argument("--dictionary", type=str, default="DICT_4X4_50", help="cv2.aruco dictionary name")
    ap.add_argument("--rig-id", type=str, default="rig_unnamed", help="Logical rig identifier stored in metadata")
    ap.add_argument("--output", required=True, help="Output .npz path for calibration artifact")
    args = ap.parse_args()

    import cv2

    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dict_id = getattr(cv2.aruco, args.dictionary, None)
    if dict_id is None:
        raise ValueError(f"unknown aruco dictionary: {args.dictionary}")

    pairs = _read_image_pairs(left_dir, right_dir)
    image_h, image_w = pairs[0][1].shape[:2]
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_length,
        args.marker_length,
        dictionary,
    )

    all_obj_points = []
    all_img_points_l = []
    all_img_points_r = []
    used_frames = []

    for name, left_img, right_img in pairs:
        corners_l, ids_l, _ = cv2.aruco.detectMarkers(left_img, dictionary)
        corners_r, ids_r, _ = cv2.aruco.detectMarkers(right_img, dictionary)
        if ids_l is None or ids_r is None:
            continue
        _, corners_l, ids_l = cv2.aruco.interpolateCornersCharuco(corners_l, ids_l, left_img, board)
        _, corners_r, ids_r = cv2.aruco.interpolateCornersCharuco(corners_r, ids_r, right_img, board)
        if ids_l is None or ids_r is None:
            continue
        id_to_corner_l = {int(i[0]): c[0] for i, c in zip(ids_l, corners_l)}
        id_to_corner_r = {int(i[0]): c[0] for i, c in zip(ids_r, corners_r)}
        common = sorted(set(id_to_corner_l).intersection(id_to_corner_r))
        if len(common) < 6:
            continue
        obj = []
        pts_l = []
        pts_r = []
        chess_corners = board.getChessboardCorners()
        for cid in common:
            obj.append(chess_corners[cid])
            pts_l.append(id_to_corner_l[cid])
            pts_r.append(id_to_corner_r[cid])
        all_obj_points.append(np.asarray(obj, dtype=np.float32))
        all_img_points_l.append(np.asarray(pts_l, dtype=np.float32))
        all_img_points_r.append(np.asarray(pts_r, dtype=np.float32))
        used_frames.append(name)

    if len(all_obj_points) < 4:
        raise RuntimeError("not enough valid stereo ChArUco detections for calibration")

    mono_rms_left, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
        all_obj_points, all_img_points_l, (image_w, image_h), None, None
    )
    mono_rms_right, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
        all_obj_points, all_img_points_r, (image_w, image_h), None, None
    )

    flags = cv2.CALIB_FIX_INTRINSIC
    stereo_err, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        all_obj_points,
        all_img_points_l,
        all_img_points_r,
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        (image_w, image_h),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=flags,
    )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, (image_w, image_h), R, T, alpha=0
    )
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(
        mtx_l, dist_l, R1, P1, (image_w, image_h), cv2.CV_32FC1
    )
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(
        mtx_r, dist_r, R2, P2, (image_w, image_h), cv2.CV_32FC1
    )
    mean_epi_err, max_epi_err = _epipolar_error(F, all_img_points_l, all_img_points_r)

    calibration_id = str(uuid.uuid4())
    np.savez_compressed(
        output_path,
        left_map_x=left_map_x,
        left_map_y=left_map_y,
        right_map_x=right_map_x,
        right_map_y=right_map_y,
        left_camera_matrix=mtx_l,
        left_dist=dist_l,
        right_camera_matrix=mtx_r,
        right_dist=dist_r,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
    )

    metadata = {
        "schema_version": "fixed_rig_calibration_v1",
        "calibration_id": calibration_id,
        "rig_id": args.rig_id,
        "image_width": image_w,
        "image_height": image_h,
        "image_size": {"width": image_w, "height": image_h},
        "board": {
            "type": "charuco",
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_size_mm": args.square_length,
            "marker_size_mm": args.marker_length,
            "dictionary": args.dictionary,
        },
        "quality": {
            "num_pairs_used": len(used_frames),
            "mono_rms_left": float(mono_rms_left),
            "mono_rms_right": float(mono_rms_right),
            "stereo_rms": float(stereo_err),
            "mean_epipolar_error_px": mean_epi_err,
            "max_epipolar_error_px": max_epi_err,
        },
        "roi_left": [int(v) for v in roi1],
        "roi_right": [int(v) for v in roi2],
        "provenance": {
            "left_dir": str(left_dir),
            "right_dir": str(right_dir),
            "used_frame_count": len(used_frames),
            "used_frames": used_frames,
            "created_at_unix": time.time(),
        },
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"wrote calibration artifact: {output_path}")
    print(f"wrote calibration metadata: {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
