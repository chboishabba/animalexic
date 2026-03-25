#!/usr/bin/env python3
"""
Tiny Vulkan dispatcher stub that mirrors dashiCORE patterns.
It compiles SPIR-V (via glslc) and outlines descriptor bindings for:
  frame_diff -> mask cleanup (optional) -> stereo_roi

This is a wiring guide; it will no-op if Vulkan is unavailable.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHADER_BUILD = REPO / "shaders" / "build"

# Try to reuse dashiCORE helpers if available
sys.path.append(str(REPO.parent / "dashiCORE"))

try:
    from gpu_vulkan_dispatcher import VulkanDispatchConfig, create_vulkan_handles  # type: ignore
    import vulkan as vk  # type: ignore
except Exception as e:  # pragma: no cover
    print(f"[stub] Vulkan not available ({e}); this stub documents bindings only.")
    sys.exit(0)


def describe_pipeline():
    print("Descriptor set 0 bindings:")
    print(" frame_diff:")
    print("  0: curr_frame r8ui")
    print("  1: prev_frame r8ui")
    print("  2: prior_mask r8ui (optional)")
    print("  3: diff_out r16ui")
    print("  4: mask_out r8ui")
    print("  5: severity_out r8ui")
    print("  6: FrameDiffParams UBO")
    print(" stereo_roi:")
    print("  0: left_r8")
    print("  1: right_r8")
    print("  2: ROITiles SSBO")
    print("  3: roi_mask_opt (optional)")
    print("  4: FrameMeta UBO")
    print("  5: StereoROIParams UBO")
    print("  6: disparity_q8")
    print("  7: cost_min")
    print("  8: conf_gap")
    print("  9: valid_mask")
    print(" 10: severity_map")
    print(" 11: prov_buf (optional SSBO)")


def main():
    if not SHADER_BUILD.exists():
        print(f"[stub] Build SPIR-V first: cmake -S . -B build && cmake --build build --target shaders")
        return

    describe_pipeline()

    # Example: create handles to validate that Vulkan is reachable.
    cfg = VulkanDispatchConfig()
    try:
        handles = create_vulkan_handles(cfg)
    except Exception as e:
        print(f"[stub] Vulkan init failed ({e}); shaders are built at {SHADER_BUILD}.")
        return

    print(f"[stub] Vulkan device ready (queue family {handles.queue_family_index}).")
    print(f"[stub] Use handles.device / handles.queue to create pipelines for:")
    for spv in ["frame_diff.spv", "stereo_roi.spv"]:
        print(f"  - {SHADER_BUILD / spv}")
    print("[stub] Actual pipeline creation and buffer binding are left to the full dispatcher.")


if __name__ == "__main__":
    main()
