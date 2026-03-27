#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from surfel_guard import SURFEL_ASCENDED, SURFEL_PLATEAU, write_colored_ply_ascii


def _state_mask(states: np.ndarray, mode: str) -> np.ndarray:
    if mode == "ascended":
        return states == SURFEL_ASCENDED
    if mode == "ascended_plateau":
        return (states == SURFEL_ASCENDED) | (states == SURFEL_PLATEAU)
    raise ValueError(f"unsupported state mode: {mode}")


def _orient_normals_toward_camera(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return normals
    out = normals.copy().astype(np.float32)
    dots = np.sum(points * out, axis=1)
    flip = dots > 0.0
    out[flip] *= -1.0
    return out


def _write_oriented_xyz(path: Path, points: np.ndarray, normals: np.ndarray, weights: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("x y z nx ny nz weight\n")
        for p, n, w in zip(points, normals, weights):
            f.write(
                f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} "
                f"{float(n[0]):.6f} {float(n[1]):.6f} {float(n[2]):.6f} "
                f"{float(w):.6f}\n"
            )


def _density_stats(densities: np.ndarray) -> dict[str, float | int]:
    if densities.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "count": int(densities.size),
        "mean": float(np.mean(densities)),
        "median": float(np.median(densities)),
        "p10": float(np.percentile(densities, 10)),
        "p90": float(np.percentile(densities, 90)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surfel-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--state-filter", choices=["ascended", "ascended_plateau"], default="ascended")
    ap.add_argument("--poisson-depth", type=int, default=8)
    ap.add_argument("--density-quantile-keep", type=float, default=0.10, help="Drop lowest-density vertices below this quantile")
    ap.add_argument("--scale", type=float, default=1.1)
    ap.add_argument("--linear-fit", action="store_true")
    args = ap.parse_args()

    state_path = args.surfel_dir / "surfels_state.npz"
    if not state_path.exists():
        raise SystemExit(f"missing surfel state: {state_path}")

    blob = np.load(state_path)
    pos = blob["pos"].astype(np.float32)
    normals = blob["normal"].astype(np.float32)
    weights = blob["weight"].astype(np.float32)
    states = blob["states"].astype(np.uint8)

    keep = _state_mask(states, args.state_filter)
    points = pos[keep]
    normals = normals[keep]
    weights = weights[keep]
    if points.size == 0:
        raise SystemExit(f"no surfels selected by state filter {args.state_filter}")

    normals = _orient_normals_toward_camera(points, normals)
    weight_norm = weights / max(float(np.max(weights)), 1e-6)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_oriented_xyz(args.output_dir / "oriented_points.xyzwn", points, normals, weight_norm)
    colors = np.clip(np.stack([255.0 * weight_norm, 220.0 * weight_norm, 80.0 + 120.0 * weight_norm], axis=1), 0, 255).astype(np.uint8)
    write_colored_ply_ascii(args.output_dir / "oriented_points_weighted.ply", points, colors)

    summary: dict[str, object] = {
        "surfel_dir": str(args.surfel_dir),
        "state_filter": args.state_filter,
        "counts": {"selected_surfels": int(len(points))},
        "params": {
            "poisson_depth": int(args.poisson_depth),
            "density_quantile_keep": float(args.density_quantile_keep),
            "scale": float(args.scale),
            "linear_fit": bool(args.linear_fit),
        },
        "artifacts": {
            "oriented_points_xyzwn": str(args.output_dir / "oriented_points.xyzwn"),
            "oriented_points_weighted_ply": str(args.output_dir / "oriented_points_weighted.ply"),
        },
    }

    try:
        import open3d as o3d  # type: ignore
    except ImportError:
        summary["open3d"] = {
            "available": False,
            "message": "open3d is not installed; wrote oriented point exports only",
        }
        (args.output_dir / "poisson_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[surfel-poisson] wrote exports only: {args.output_dir}")
        print("[surfel-poisson] open3d not installed; skip mesh reconstruction")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(args.poisson_depth),
        scale=float(args.scale),
        linear_fit=bool(args.linear_fit),
    )
    densities_np = np.asarray(densities, dtype=np.float32)
    mesh_path = args.output_dir / "poisson_mesh.ply"
    density_mesh_path = args.output_dir / "poisson_mesh_filtered.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    if densities_np.size:
        cutoff = float(np.quantile(densities_np, float(args.density_quantile_keep)))
        keep_vertices = densities_np >= cutoff
        mesh_filtered = mesh.remove_vertices_by_mask(~keep_vertices)
        o3d.io.write_triangle_mesh(str(density_mesh_path), mesh_filtered)
        filtered_vertices = int(np.count_nonzero(keep_vertices))
    else:
        cutoff = 0.0
        filtered_vertices = 0

    summary["open3d"] = {
        "available": True,
        "mesh_path": str(mesh_path),
        "filtered_mesh_path": str(density_mesh_path),
        "mesh_vertices": int(np.asarray(mesh.vertices).shape[0]),
        "mesh_triangles": int(np.asarray(mesh.triangles).shape[0]),
        "filtered_vertices_kept": filtered_vertices,
        "density_cutoff": cutoff,
        "density_stats": _density_stats(densities_np),
    }
    (args.output_dir / "poisson_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[surfel-poisson] wrote exports: {args.output_dir / 'oriented_points.xyzwn'}")
    print(f"[surfel-poisson] wrote mesh: {mesh_path}")
    print(f"[surfel-poisson] wrote filtered mesh: {density_mesh_path}")


if __name__ == "__main__":
    main()
