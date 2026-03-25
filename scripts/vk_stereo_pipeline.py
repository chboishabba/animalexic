#!/usr/bin/env python3
"""
Minimal Vulkan pipeline wiring for frame_diff -> stereo_roi.

Notes:
- Designed to mirror the descriptor layouts in shaders/frame_diff.comp and shaders/stereo_roi.comp.
- Uses dashiCORE's create_vulkan_handles helper for device/queue selection.
- Safe to import and dry-run on machines without Vulkan; exits with a friendly message.
- This is intentionally slim: no command-line options, no full image IO; it just
  builds pipelines and descriptor layouts and prints what would be dispatched.
"""

from pathlib import Path
from typing import Optional
import sys

REPO = Path(__file__).resolve().parents[1]
SHADERS = REPO / "shaders" / "build"

sys.path.append(str(REPO.parent / "dashiCORE"))

try:
    from gpu_vulkan_dispatcher import VulkanDispatchConfig, create_vulkan_handles  # type: ignore
    import vulkan as vk  # type: ignore
except Exception as e:  # pragma: no cover
    print(f"[vk-stereo] Vulkan unavailable: {e}")
    sys.exit(0)


def load_spv(path: Path):
    with path.open("rb") as f:
        return f.read()


def create_shader_module(device, code: bytes):
    info = vk.VkShaderModuleCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        codeSize=len(code),
        pCode=code,
    )
    return vk.vkCreateShaderModule(device, info, None)


def create_descriptor_set_layout(device, bindings):
    layout_bindings = []
    for b in bindings:
        layout_bindings.append(
            vk.VkDescriptorSetLayoutBinding(
                binding=b["binding"],
                descriptorType=b["type"],
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers=None,
            )
        )
    info = vk.VkDescriptorSetLayoutCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        bindingCount=len(layout_bindings),
        pBindings=layout_bindings,
    )
    return vk.vkCreateDescriptorSetLayout(device, info, None)


def create_pipeline_layout(device, set_layout):
    info = vk.VkPipelineLayoutCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        setLayoutCount=1,
        pSetLayouts=[set_layout],
        pushConstantRangeCount=0,
    )
    return vk.vkCreatePipelineLayout(device, info, None)


def create_compute_pipeline(device, shader_module, entry_point, pipeline_layout):
    stage = vk.VkPipelineShaderStageCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        module=shader_module,
        pName=entry_point,
    )
    info = vk.VkComputePipelineCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        stage=stage,
        layout=pipeline_layout,
    )
    return vk.vkCreateComputePipelines(device, vk.VK_NULL_HANDLE, 1, [info], None)[0]


def bindings_frame_diff():
    return [
        {"binding": 0, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},  # curr_frame
        {"binding": 1, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},  # prev_frame
        {"binding": 2, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},  # prior_mask
        {"binding": 3, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},  # diff_out
        {"binding": 4, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},  # mask_out
        {"binding": 5, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},  # severity_out
        {"binding": 6, "type": vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},  # FrameDiffParams
    ]


def bindings_stereo_roi():
    return [
        {"binding": 0, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},  # left_r8
        {"binding": 1, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},  # right_r8
        {"binding": 2, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},  # ROITiles
        {"binding": 3, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},  # roi_mask_opt
        {"binding": 4, "type": vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},  # FrameMeta
        {"binding": 5, "type": vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},  # StereoROIParams
        {"binding": 6, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},  # disparity_q8
        {"binding": 7, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},  # cost_min
        {"binding": 8, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},  # conf_gap
        {"binding": 9, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},  # valid_mask
        {"binding": 10, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},  # severity_map
        {"binding": 11, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},  # prov_buf
    ]


def main():
    if not SHADERS.exists():
        print("[vk-stereo] SPV missing; build with cmake --build build --target shaders")
        sys.exit(1)

    cfg = VulkanDispatchConfig()
    try:
        handles = create_vulkan_handles(cfg)
    except Exception as e:
        print(f"[vk-stereo] Vulkan init failed ({e}); cannot dispatch here.")
        sys.exit(0)

    device = handles.device

    # frame_diff pipeline
    spv_fd = load_spv(SHADERS / "frame_diff.spv")
    mod_fd = create_shader_module(device, spv_fd)
    layout_fd = create_descriptor_set_layout(device, bindings_frame_diff())
    pl_layout_fd = create_pipeline_layout(device, layout_fd)
    pipe_fd = create_compute_pipeline(device, mod_fd, b"main", pl_layout_fd)

    # stereo_roi pipeline
    spv_sr = load_spv(SHADERS / "stereo_roi.spv")
    mod_sr = create_shader_module(device, spv_sr)
    layout_sr = create_descriptor_set_layout(device, bindings_stereo_roi())
    pl_layout_sr = create_pipeline_layout(device, layout_sr)
    pipe_sr = create_compute_pipeline(device, mod_sr, b"main", pl_layout_sr)

    print("[vk-stereo] Pipelines created:")
    print(f"  frame_diff -> {SHADERS / 'frame_diff.spv'}")
    print(f"  stereo_roi -> {SHADERS / 'stereo_roi.spv'}")
    print("[vk-stereo] Descriptor set layouts mirror shader bindings.")
    print("[vk-stereo] You can now allocate images/SSBOs, bind descriptor sets, and dispatch:")
    print("  - Dispatch frame_diff over full frame grid")
    print("  - (Optional) run mask cleanup via core_mask")
    print("  - Dispatch stereo_roi over ROI tiles (one workgroup per tile)")

    # Cleanup
    vk.vkDestroyPipeline(device, pipe_fd, None)
    vk.vkDestroyPipeline(device, pipe_sr, None)
    vk.vkDestroyPipelineLayout(device, pl_layout_fd, None)
    vk.vkDestroyPipelineLayout(device, pl_layout_sr, None)
    vk.vkDestroyDescriptorSetLayout(device, layout_fd, None)
    vk.vkDestroyDescriptorSetLayout(device, layout_sr, None)
    vk.vkDestroyShaderModule(device, mod_fd, None)
    vk.vkDestroyShaderModule(device, mod_sr, None)
    handles.close()


if __name__ == "__main__":
    main()
