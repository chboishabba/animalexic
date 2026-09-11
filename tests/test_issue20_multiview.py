import json
import tempfile
import unittest
from pathlib import Path

from scripts.issue20_multiview import (
    MetadataError,
    accumulate_sparse_motion_voxels,
    group_synchronised_frames,
    load_issue20_metadata,
    pixel_ray_world,
)


def entry(cam, frame, pos, yaw=0.0, pitch=0.0, roll=0.0, fov=60.0, image=None):
    return {
        "camera_index": cam,
        "frame_index": frame,
        "camera_position": list(pos),
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "fov_degrees": fov,
        "image_file": image or f"cam{cam}_{frame:04d}.png",
    }


class Issue20MetadataTests(unittest.TestCase):
    def write_metadata(self, rows):
        td = tempfile.TemporaryDirectory()
        path = Path(td.name) / "frame_metadata.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        self.addCleanup(td.cleanup)
        return path

    def test_loads_strict_known_pose_metadata(self):
        path = self.write_metadata([
            entry(0, 0, (-1.0, 0.0, 0.0)),
            entry(1, 0, (1.0, 0.0, 0.0)),
        ])
        obs = load_issue20_metadata(path)
        self.assertEqual(len(obs), 2)
        self.assertEqual(obs[0].camera_id, 0)
        self.assertEqual(obs[1].position, (1.0, 0.0, 0.0))
        self.assertEqual(obs[0].pose_source, "issue20_known_pose")

    def test_missing_pose_fails_closed(self):
        row = entry(0, 0, (0.0, 0.0, 0.0))
        del row["camera_position"]
        path = self.write_metadata([row])
        with self.assertRaises(MetadataError):
            load_issue20_metadata(path)

    def test_group_requires_all_requested_cameras(self):
        path = self.write_metadata([
            entry(0, 0, (-1, 0, 0)), entry(1, 0, (1, 0, 0)),
            entry(0, 1, (-1, 0, 0)),
        ])
        obs = load_issue20_metadata(path)
        groups = group_synchronised_frames(obs, required_camera_ids={0, 1})
        self.assertEqual(list(groups), [0])
        self.assertEqual({o.camera_id for o in groups[0]}, {0, 1})

    def test_center_pixel_ray_respects_source_rotation_convention(self):
        path = self.write_metadata([entry(0, 0, (0, 0, 0), yaw=90.0)])
        camera = load_issue20_metadata(path)[0]
        ray = pixel_ray_world(camera, u=50, v=50, width=100, height=100)
        self.assertAlmostEqual(ray[0], 0.0, places=6)
        self.assertAlmostEqual(ray[1], 0.0, places=6)
        self.assertAlmostEqual(ray[2], 1.0, places=6)

    def test_sparse_motion_from_two_cameras_accumulates_shared_voxel(self):
        path = self.write_metadata([
            entry(0, 0, (-1.0, 0.0, 0.0)),
            entry(1, 0, (1.0, 0.0, 0.0)),
        ])
        cams = {o.camera_id: o for o in load_issue20_metadata(path)}
        samples = [
            {"camera_id": 0, "frame_index": 0, "direction": (1.0, 0.0, 5.0), "evidence": 1.0},
            {"camera_id": 1, "frame_index": 0, "direction": (-1.0, 0.0, 5.0), "evidence": 1.0},
        ]
        voxels = accumulate_sparse_motion_voxels(
            cameras=cams,
            samples=samples,
            voxel_size=1.0,
            max_distance=8.0,
            min_camera_support=2,
        )
        shared_near_intersection = [
            (key, cell) for key, cell in voxels.items()
            if abs(key[0]) <= 1 and abs(key[1]) <= 1 and key[2] in {4, 5}
        ]
        self.assertTrue(shared_near_intersection)
        self.assertTrue(all(cell["camera_support"] >= 2 for _, cell in shared_near_intersection))
        self.assertTrue(all(cell["status"] == "candidate" for _, cell in shared_near_intersection))


if __name__ == "__main__":
    unittest.main()
