#!/usr/bin/env python3
"""
End-to-end stereo runtime:
- streams frames from a YouTube URL or local file via ffmpeg (no full download)
- optional side-by-side split for 3D videos
- can load a fixed-rig calibration artifact and rectify each pair
- builds delta ROI tiles, computes CPU census stereo candidates, promotes canonical disparity,
  and writes append-only receipts to SQLite
- if Vulkan is available, prints pipeline readiness (reuses vk_stereo_pipeline bindings);
  GPU dispatch is left as a TODO hook.

Usage examples:
  python scripts/run_stereo_dispatch.py --youtube https://youtu.be/... --width 640 --height 360 --every-n 2 --sbs
  python scripts/run_stereo_dispatch.py --file /path/to/video.mp4 --width 640 --height 360
"""

import argparse
import csv
import subprocess
import threading
import collections
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from fixed_rig_runtime import (
    ReceiptStore,
    build_delta_roi,
    depth_from_disparity,
    load_calibration_artifact,
    merge_disparity_state,
    promote_disparity,
    rectify_pair,
    TemporalMergeParams,
)

REPO = Path(__file__).resolve().parents[1]
SHADERS = REPO / "shaders" / "build"
sys.path.append(str(REPO.parent / "dashiCORE"))

try:
    from gpu_vulkan_dispatcher import VulkanDispatchConfig, create_vulkan_handles  # type: ignore
    from gpu_common_methods import find_memory_type  # type: ignore
    import vulkan as vk  # type: ignore
    HAS_VULKAN = True
except Exception:
    HAS_VULKAN = False


def vk_require():
    if not HAS_VULKAN:
        raise RuntimeError("Vulkan not available")
    # constants from dispatcher
    global HOST_VISIBLE_COHERENT
    if "HOST_VISIBLE_COHERENT" not in globals():
        HOST_VISIBLE_COHERENT = (1 << 1) | (1 << 2)


def vk_fmt_channels(fmt: int) -> int:
    if fmt in (vk.VK_FORMAT_R8_UINT, vk.VK_FORMAT_R8_UNORM):
        return 1
    if fmt in (vk.VK_FORMAT_R16_UINT,):
        return 1
    raise ValueError(f"unsupported format {fmt}")


def create_image(handles, width, height, fmt, usage):
    """Create a host-visible linear image for simplicity."""
    vk_require()
    width = int(width)
    height = int(height)
    ci = vk.VkImageCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        imageType=vk.VK_IMAGE_TYPE_2D,
        format=fmt,
        extent=vk.VkExtent3D(width=width, height=height, depth=1),
        mipLevels=1,
        arrayLayers=1,
        samples=vk.VK_SAMPLE_COUNT_1_BIT,
        tiling=vk.VK_IMAGE_TILING_LINEAR,
        usage=usage,
        sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        initialLayout=vk.VK_IMAGE_LAYOUT_PREINITIALIZED,
    )
    img = vk.vkCreateImage(handles.device, ci, None)
    mem_req = vk.vkGetImageMemoryRequirements(handles.device, img)
    mem_type_index = find_memory_type(handles.mem_props, mem_req.memoryTypeBits, HOST_VISIBLE_COHERENT)
    alloc = vk.VkMemoryAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=mem_req.size,
        memoryTypeIndex=mem_type_index,
    )
    mem = vk.vkAllocateMemory(handles.device, alloc, None)
    vk.vkBindImageMemory(handles.device, img, mem, 0)
    return img, mem, mem_req.size


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


def create_shader_module(device, code: bytes):
    info = vk.VkShaderModuleCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        codeSize=len(code),
        pCode=code,
    )
    return vk.vkCreateShaderModule(device, info, None)


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


def make_view(device, img, fmt):
    ci = vk.VkImageViewCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        image=img,
        viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
        format=fmt,
        components=vk.VkComponentMapping(r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                                         g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                                         b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                                         a=vk.VK_COMPONENT_SWIZZLE_IDENTITY),
        subresourceRange=vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        ),
    )
    return vk.vkCreateImageView(device, ci, None)


def _write_buffer(device, memory, data_arr: np.ndarray):
    data_bytes = data_arr.tobytes()
    mapped = vk.vkMapMemory(device, memory, 0, len(data_bytes), 0)
    try:
        mv = memoryview(mapped)
        mv[: len(data_bytes)] = data_bytes
    finally:
        vk.vkUnmapMemory(device, memory)


def _create_buffer(device, mem_props, size: int, usage: int, required_flags: int):
    ci = vk.VkBufferCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        size=int(size),
        usage=usage,
        sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
    )
    buf = vk.vkCreateBuffer(device, ci, None)
    mem_req = vk.vkGetBufferMemoryRequirements(device, buf)
    mem_type_index = find_memory_type(mem_props, mem_req.memoryTypeBits, required_flags)
    alloc = vk.VkMemoryAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=mem_req.size,
        memoryTypeIndex=mem_type_index,
    )
    mem = vk.vkAllocateMemory(device, alloc, None)
    vk.vkBindBufferMemory(device, buf, mem, 0)
    return buf, mem


class FrameStreamer:
    def __init__(self, src: str, width: int, height: int, every_n: int, gray: bool, youtube: bool, start_seconds: float = 0.0):
        self.width = width
        self.height = height
        self.gray = gray
        pix_fmt = "gray" if gray else "rgb24"
        vf = f"select='not(mod(n\\,{every_n}))',scale={width}:{height},format={pix_fmt}"
        input_arg = src if not youtube else self._youtube_best(src)
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
        ]
        if start_seconds > 0:
            cmd.extend(["-ss", str(start_seconds)])
        cmd.extend([
            "-i",
            input_arg,
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-",
        ])
        self.frame_bytes = width * height * (1 if gray else 3)
        self.buf = collections.deque(maxlen=30)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 6)
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _youtube_best(self, url: str) -> str:
        import yt_dlp

        ydl = yt_dlp.YoutubeDL(
            {
                "quiet": True,
                "format": "bestvideo[ext=mp4][vcodec~='(avc1|h264)']/best",
                "extractor_args": {"youtube": {"player_client": ["android"]}},
            }
        )
        info = ydl.extract_info(url, download=False)
        return info["url"]

    def _reader(self):
        count = 0
        while True:
            raw = self.proc.stdout.read(self.frame_bytes)
            if len(raw) != self.frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8)
            if self.gray:
                frame = frame.reshape(self.height, self.width)
            else:
                frame = frame.reshape(self.height, self.width, 3)
            self.buf.append(frame)
            count += 1
        print(f"[stream] reader stopped after {count} frames")

    def get_pair(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if len(self.buf) < 2:
            return None, None
        return self.buf[-2], self.buf[-1]

    def get_latest(self) -> Optional[np.ndarray]:
        if not self.buf:
            return None
        return self.buf[-1]

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()


def census5x5(img: np.ndarray, x: int, y: int) -> int:
    center = img[y, x]
    bits = 0
    bit = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            if img[y + dy, x + dx] < center:
                bits |= 1 << bit
            bit += 1
    return bits


def stereo_census_roi(
    left: np.ndarray,
    right: np.ndarray,
    d_min=0,
    d_max=64,
    min_tex=8,
    max_cost=30,
    min_conf=2,
    roi_mask: Optional[np.ndarray] = None,
):
    h, w = left.shape
    disp = np.zeros((h, w), dtype=np.uint16)
    cost_min = np.full((h, w), 0xFFFF, dtype=np.uint16)
    conf = np.zeros((h, w), dtype=np.uint16)
    valid = np.zeros((h, w), dtype=np.uint8)
    for y in range(2, h - 2):
        for x in range(2 + d_max, w - 2):
            if roi_mask is not None and roi_mask[y, x] == 0:
                continue
            patch = left[y - 2 : y + 3, x - 2 : x + 3]
            if patch.max() - patch.min() < min_tex:
                continue
            cL = census5x5(left, x, y)
            best = 1 << 30
            second = 1 << 30
            best_d = 0
            for d in range(d_min, d_max + 1):
                xr = x - d
                if xr < 2:
                    continue
                cR = census5x5(right, xr, y)
                c = bin(cL ^ cR).count("1")
                if c < best:
                    second = best
                    best = c
                    best_d = d
                elif c < second:
                    second = c
            if best == (1 << 30):
                continue
            cg = 0 if second == (1 << 30) else (second - best)
            if best <= max_cost and cg >= min_conf:
                disp[y, x] = best_d << 8
                cost_min[y, x] = best
                conf[y, x] = cg
                valid[y, x] = 1
    return disp, cost_min, conf, valid


def motion_roi_mask(prev_frame: np.ndarray, curr_frame: np.ndarray, diff_threshold: int = 12, min_luma: int = 5):
    diff = np.abs(curr_frame.astype(np.int16) - prev_frame.astype(np.int16)).astype(np.uint16)
    bright = np.maximum(prev_frame, curr_frame) >= min_luma
    mask = ((diff >= diff_threshold) & bright).astype(np.uint8)
    return diff, mask


def save_png(arr: np.ndarray, path: Path, scale: float = 1.0):
    if arr.dtype != np.uint8:
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def print_cpu_stats(disp: np.ndarray, cost_min: np.ndarray, conf: np.ndarray, valid: np.ndarray):
    valid_count = int(valid.sum())
    total = int(valid.size)
    coverage = (100.0 * valid_count / total) if total else 0.0
    print(f"[run] valid pixels: {valid_count}/{total} ({coverage:.2f}%)")
    if valid_count == 0:
        print("[run] disparity stats: no valid disparity pixels")
        return
    disp_px = (disp[valid != 0] >> 8).astype(np.uint16)
    costs = cost_min[valid != 0].astype(np.uint16)
    gaps = conf[valid != 0].astype(np.uint16)
    print(
        "[run] disparity stats:"
        f" min={int(disp_px.min())}"
        f" max={int(disp_px.max())}"
        f" mean={float(disp_px.mean()):.2f}"
        f" cost_mean={float(costs.mean()):.2f}"
        f" conf_mean={float(gaps.mean()):.2f}"
    )


def _parse_csv_int_list(value: str, name: str):
    values = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(f"invalid {name} value: {token}") from exc
    if not values:
        raise ValueError(f"no values parsed for {name}")
    return values


def _wait_for_frame(streamer: FrameStreamer, args, last_seen):
    import time

    t0 = time.time()
    while time.time() - t0 < args.timeout:
        latest = streamer.get_latest()
        if latest is None or latest is last_seen:
            threading.Event().wait(0.03)
            continue
        if float(latest.mean()) < args.min_frame_mean:
            threading.Event().wait(0.03)
            continue
        if args.sbs:
            return latest, latest, latest
        left, right = streamer.get_pair()
        if left is None or right is None:
            threading.Event().wait(0.03)
            continue
        return right, left, right
    return None, None, None


def _save_frame_outputs(outdir: Path, frame_index: int, left: np.ndarray, right: np.ndarray, diff: np.ndarray, roi_mask: np.ndarray, candidate: np.ndarray, canonical: np.ndarray, promoted_mask: np.ndarray, depth: Optional[np.ndarray]):
    depth_vis = None
    if depth is not None:
        valid = depth > 0
        depth_vis = np.zeros_like(depth, dtype=np.float32)
        if np.any(valid):
            depth_vis[valid] = depth[valid] / max(1e-6, float(depth[valid].max())) * 255.0
    save_png(left.astype(np.uint8), outdir / f"left_input_f{frame_index:04d}.png")
    save_png(right.astype(np.uint8), outdir / f"right_input_f{frame_index:04d}.png")
    save_png(np.clip(diff, 0, 255).astype(np.uint8), outdir / f"motion_diff_f{frame_index:04d}.png")
    save_png(roi_mask * 255, outdir / f"roi_mask_f{frame_index:04d}.png")
    save_png((candidate >> 8).astype(np.uint8), outdir / f"candidate_disp_f{frame_index:04d}.png", scale=4)
    save_png((canonical >> 8).astype(np.uint8), outdir / f"canonical_disp_f{frame_index:04d}.png", scale=4)
    save_png(promoted_mask * 255, outdir / f"promoted_mask_f{frame_index:04d}.png")
    if depth_vis is not None:
        save_png(depth_vis, outdir / f"depth_f{frame_index:04d}.png", scale=1.0)
    save_png(left.astype(np.uint8), outdir / "left_input.png")
    save_png(right.astype(np.uint8), outdir / "right_input.png")
    save_png(roi_mask * 255, outdir / "roi_mask.png")
    save_png((candidate >> 8).astype(np.uint8), outdir / "disp_cpu.png", scale=4)
    save_png((canonical >> 8).astype(np.uint8), outdir / "canonical_disp.png", scale=4)
    save_png(promoted_mask * 255, outdir / "valid_cpu.png")
    if depth_vis is not None:
        save_png(depth_vis, outdir / "depth.png", scale=1.0)


def run_vulkan(left: np.ndarray, right: np.ndarray):
    """Minimal GPU dispatch: upload left/right, run stereo_roi only, read disparity."""
    print("[run][gpu] starting Vulkan dispatch")
    vk_require()
    handles = create_vulkan_handles(VulkanDispatchConfig())
    device = handles.device

    h, w = left.shape
    # Formats
    fmt_r8 = vk.VK_FORMAT_R8_UINT
    fmt_r16 = vk.VK_FORMAT_R16_UINT

    # Images
    img_left, mem_left, mem_left_size = create_image(handles, w, h, fmt_r8, vk.VK_IMAGE_USAGE_STORAGE_BIT)
    img_right, mem_right, mem_right_size = create_image(handles, w, h, fmt_r8, vk.VK_IMAGE_USAGE_STORAGE_BIT)
    img_disp, mem_disp, mem_disp_size = create_image(handles, w, h, fmt_r16, vk.VK_IMAGE_USAGE_STORAGE_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
    img_cost, mem_cost, mem_cost_size = create_image(handles, w, h, fmt_r16, vk.VK_IMAGE_USAGE_STORAGE_BIT)
    img_conf, mem_conf, mem_conf_size = create_image(handles, w, h, fmt_r16, vk.VK_IMAGE_USAGE_STORAGE_BIT)
    img_valid, mem_valid, mem_valid_size = create_image(handles, w, h, fmt_r8, vk.VK_IMAGE_USAGE_STORAGE_BIT)
    img_sev, mem_sev, mem_sev_size = create_image(handles, w, h, fmt_r8, vk.VK_IMAGE_USAGE_STORAGE_BIT)

    view_left = make_view(device, img_left, fmt_r8)
    view_right = make_view(device, img_right, fmt_r8)
    view_disp = make_view(device, img_disp, fmt_r16)
    view_cost = make_view(device, img_cost, fmt_r16)
    view_conf = make_view(device, img_conf, fmt_r16)
    view_valid = make_view(device, img_valid, fmt_r8)
    view_sev = make_view(device, img_sev, fmt_r8)

    # Write left/right into linear images
    def write_image(mem, mem_size, fmt, data):
        mapped = vk.vkMapMemory(device, mem, 0, mem_size, 0)
        try:
            arr = np.array(data, copy=False)
            ch = vk_fmt_channels(fmt)
            if ch == 1:
                buf = arr.astype(np.uint8 if fmt == fmt_r8 else np.uint16).tobytes()
            else:
                buf = arr.tobytes()
            mv = memoryview(mapped)
            mv[: len(buf)] = buf
        finally:
            vk.vkUnmapMemory(device, mem)

    write_image(mem_left, mem_left_size, fmt_r8, left)
    write_image(mem_right, mem_right_size, fmt_r8, right)

    # ROI tiles covering full image with 16x16 tiles
    tile_w = 16
    tile_h = 16
    tiles = []
    for y0 in range(0, h, tile_h):
        for x0 in range(0, w, tile_w):
            tiles.append((x0, y0, min(tile_w, w - x0), min(tile_h, h - y0)))
    tiles_np = np.array(tiles, dtype=np.uint32)
    tiles_buf, tiles_mem = _create_buffer(device, handles.mem_props, tiles_np.nbytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST_VISIBLE_COHERENT)
    _write_buffer(device, tiles_mem, tiles_np.astype(np.uint32, copy=False))

    # UBOs
    frame_meta = np.array([w, h, w, w, tile_w, tile_h, len(tiles), 0], dtype=np.uint32)
    meta_buf, meta_mem = _create_buffer(device, handles.mem_props, frame_meta.nbytes, vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, HOST_VISIBLE_COHERENT)
    _write_buffer(device, meta_mem, frame_meta.astype(np.uint32, copy=False))

    params = np.array([0, 64, 2, 0, 8, 30, 2, 2, 2, 0, 0, 0, 0, 4, 3], dtype=np.uint32)
    # pad to 16*? ensure length multiple of 4 bytes; already np.uint32
    params_buf, params_mem = _create_buffer(device, handles.mem_props, params.nbytes, vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, HOST_VISIBLE_COHERENT)
    _write_buffer(device, params_mem, params.astype(np.uint32, copy=False))

    # Descriptor set layout
    bindings = [
        {"binding": 0, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
        {"binding": 1, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
        {"binding": 2, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {"binding": 3, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
        {"binding": 4, "type": vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
        {"binding": 5, "type": vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
        {"binding": 6, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
        {"binding": 7, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
        {"binding": 8, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
        {"binding": 9, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
        {"binding": 10, "type": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
    ]
    ds_layout = create_descriptor_set_layout(device, bindings)
    pipeline_layout = create_pipeline_layout(device, ds_layout)
    # Pipeline
    spv_sr = (SHADERS / "stereo_roi.spv").read_bytes()
    shader_module = create_shader_module(device, spv_sr)
    pipeline = create_compute_pipeline(device, shader_module, b"main", pipeline_layout)

    # Descriptor pool
    pool_sizes = [
        vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, descriptorCount=10),
        vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=4),
        vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=4),
    ]
    pool_info = vk.VkDescriptorPoolCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        maxSets=1,
        poolSizeCount=len(pool_sizes),
        pPoolSizes=pool_sizes,
    )
    desc_pool = vk.vkCreateDescriptorPool(device, pool_info, None)
    alloc_info = vk.VkDescriptorSetAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        descriptorPool=desc_pool,
        descriptorSetCount=1,
        pSetLayouts=[ds_layout],
    )
    desc_set = vk.vkAllocateDescriptorSets(device, alloc_info)[0]

    def img_info(view, layout=vk.VK_IMAGE_LAYOUT_GENERAL):
        return vk.VkDescriptorImageInfo(
            sampler=vk.VK_NULL_HANDLE,
            imageView=view,
            imageLayout=layout,
        )

    writes = [
        vk.VkWriteDescriptorSet(sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                dstSet=desc_set, dstBinding=0, descriptorCount=1,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                pImageInfo=[img_info(view_left)]),
        vk.VkWriteDescriptorSet(sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                dstSet=desc_set, dstBinding=1, descriptorCount=1,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                pImageInfo=[img_info(view_right)]),
        vk.VkWriteDescriptorSet(sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                dstSet=desc_set, dstBinding=2, descriptorCount=1,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=tiles_buf, offset=0, range=tiles_np.nbytes)]),
        vk.VkWriteDescriptorSet(sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                dstSet=desc_set, dstBinding=3, descriptorCount=1,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                pImageInfo=[img_info(view_valid)]),  # reuse valid as dummy roi mask
        vk.VkWriteDescriptorSet(sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                dstSet=desc_set, dstBinding=4, descriptorCount=1,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=meta_buf, offset=0, range=frame_meta.nbytes)]),
        vk.VkWriteDescriptorSet(sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                dstSet=desc_set, dstBinding=5, descriptorCount=1,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=params_buf, offset=0, range=params.nbytes)]),
        # outputs
        vk.VkWriteDescriptorSet(dstSet=desc_set, dstBinding=6, descriptorCount=1,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                pImageInfo=[img_info(view_disp)]),
        vk.VkWriteDescriptorSet(dstSet=desc_set, dstBinding=7, descriptorCount=1,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                pImageInfo=[img_info(view_cost)]),
        vk.VkWriteDescriptorSet(dstSet=desc_set, dstBinding=8, descriptorCount=1,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                pImageInfo=[img_info(view_conf)]),
        vk.VkWriteDescriptorSet(dstSet=desc_set, dstBinding=9, descriptorCount=1,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                pImageInfo=[img_info(view_valid)]),
        vk.VkWriteDescriptorSet(dstSet=desc_set, dstBinding=10, descriptorCount=1,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                pImageInfo=[img_info(view_sev)]),
    ]
    vk.vkUpdateDescriptorSets(device, len(writes), writes, 0, None)

    # Command buffer
    command_pool_info = vk.VkCommandPoolCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        queueFamilyIndex=handles.queue_family_index,
    )
    command_pool = vk.vkCreateCommandPool(device, command_pool_info, None)
    alloc = vk.VkCommandBufferAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        commandPool=command_pool,
        level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount=1,
    )
    cmd = vk.vkAllocateCommandBuffers(device, alloc)[0]

    begin_info = vk.VkCommandBufferBeginInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vk.vkBeginCommandBuffer(cmd, begin_info)

    # Bind pipeline
    vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
    vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, [desc_set], 0, None)

    vk.vkCmdDispatch(cmd, len(tiles), 1, 1)

    vk.vkEndCommandBuffer(cmd)

    # Submit
    submit_info = vk.VkSubmitInfo(
        sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
        commandBufferCount=1,
        pCommandBuffers=[cmd],
    )
    vk.vkQueueSubmit(handles.queue, 1, [submit_info], vk.VK_NULL_HANDLE)
    vk.vkQueueWaitIdle(handles.queue)

    # Read disparity (host-visible linear)
    mapped = vk.vkMapMemory(device, mem_disp, 0, mem_disp_size, 0)
    disp_bytes = memoryview(mapped)[: w * h * 2]
    disp = np.frombuffer(disp_bytes, dtype=np.uint16).reshape(h, w)
    vk.vkUnmapMemory(device, mem_disp)
    save_png((disp >> 8).astype(np.uint8), Path("outputs/disp_gpu.png"), scale=4)
    print("[run][gpu] saved outputs/disp_gpu.png")

    # Cleanup
    for view in (view_left, view_right, view_disp, view_cost, view_conf, view_valid, view_sev):
        vk.vkDestroyImageView(device, view, None)
    for img in (img_left, img_right, img_disp, img_cost, img_conf, img_valid, img_sev):
        vk.vkDestroyImage(device, img, None)
    for mem in (mem_left, mem_right, mem_disp, mem_cost, mem_conf, mem_valid, mem_sev):
        vk.vkFreeMemory(device, mem, None)
    vk.vkDestroyBuffer(device, tiles_buf, None)
    vk.vkFreeMemory(device, tiles_mem, None)
    vk.vkDestroyBuffer(device, meta_buf, None)
    vk.vkFreeMemory(device, meta_mem, None)
    vk.vkDestroyBuffer(device, params_buf, None)
    vk.vkFreeMemory(device, params_mem, None)
    vk.vkDestroyDescriptorPool(device, desc_pool, None)
    vk.vkDestroyPipeline(device, pipeline, None)
    vk.vkDestroyPipelineLayout(device, pipeline_layout, None)
    vk.vkDestroyDescriptorSetLayout(device, ds_layout, None)
    vk.vkDestroyShaderModule(device, shader_module, None)
    vk.vkDestroyCommandPool(device, command_pool, None)
    handles.close()
    print("[run][gpu] done")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--youtube", help="YouTube URL (SBS accepted if --sbs)")
    src.add_argument("--file", help="Local video file path (SBS accepted if --sbs)")
    ap.add_argument("--left-file", type=str, help="Local left video file path (use with --right-file for dual-file fixed rig)")
    ap.add_argument("--right-file", type=str, help="Local right video file path (use with --left-file for dual-file fixed rig)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--every-n", type=int, default=2, help="Subsample frames")
    ap.add_argument("--sbs", action="store_true", help="Source is side-by-side stereo; split into L/R")
    ap.add_argument("--output-dir", type=str, default="outputs", help="Where to save disparity PNGs")
    ap.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for first frames before giving up")
    ap.add_argument("--start-seconds", type=float, default=0.0, help="Seek this many seconds into the source before extracting frames")
    ap.add_argument("--calibration", type=str, help="Path to fixed-rig calibration artifact (.npz)")
    ap.add_argument("--receipts-db", type=str, default="outputs/receipts.sqlite", help="SQLite path for append-only receipts")
    ap.add_argument("--max-frames", type=int, default=1, help="How many frames to process in the governed runtime loop")
    ap.add_argument("--tile-size", type=int, default=16, help="Tile size for ROI scheduling")
    ap.add_argument("--tile-halo", type=int, default=1, help="Tile halo expansion for ROI scheduling")
    ap.add_argument(
        "--merge-profile",
        choices=["demo", "demo_loose", "calibrated", "strict"],
        default="strict",
        help="Temporal merge preset",
    )
    ap.add_argument("--gpu", action="store_true", help="Attempt GPU dispatch (Vulkan) after CPU")
    ap.add_argument("--min-frame-mean", type=float, default=5.0, help="Wait until frame mean exceeds this before locking input")
    ap.add_argument("--sweep", action="store_true", help="Evaluate stereo matcher threshold combinations and report summary")
    ap.add_argument("--sweep-min-tex", default="4,8", help="Comma-separated min_tex values for sweep")
    ap.add_argument("--sweep-max-cost", default="24,30,48", help="Comma-separated max_cost values for sweep")
    ap.add_argument("--sweep-min-conf", default="0,1,2", help="Comma-separated min_conf values for sweep")
    ap.add_argument("--debug-loose", action="store_true", help="Relax matcher thresholds for coverage/debugging")
    ap.add_argument("--motion-roi", action="store_true", help="Use frame-diff ROI gating for non-SBS sources")
    ap.add_argument("--diff-threshold", type=int, default=None, help="Frame diff threshold for motion ROI (defaults vary by profile)")
    args = ap.parse_args()
    if not (args.youtube or args.file or (args.left_file and args.right_file)):
        ap.error("provide --youtube, --file, or both --left-file and --right-file")

    if args.left_file or args.right_file:
        if not (args.left_file and args.right_file):
            print("[run] specify both --left-file and --right-file for dual-file mode")
            sys.exit(1)
        if args.youtube or args.file:
            print("[run] dual-file mode is exclusive of --youtube/--file")
            sys.exit(1)
        src_path = f"{args.left_file}|{args.right_file}"
        dual_file_mode = True
    else:
        src_path = args.youtube or args.file
        dual_file_mode = False
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration_artifact(Path(args.calibration)) if args.calibration else None
    receipts = ReceiptStore(Path(args.receipts_db))
    if calibration is not None:
        print(f"[run] loaded calibration artifact {calibration.calibration_id}")
    else:
        print("[run] no calibration artifact provided; running unrectified demo mode")
    # Merge profile presets
    if args.merge_profile == "demo":
        merge_params = TemporalMergeParams(
            tau_close_disp=1.1,
            max_age_keep=24,
            smooth_alpha_strong=0.75,
            smooth_alpha_weak=0.45,
        )
        default_diff_threshold = 20
    elif args.merge_profile == "demo_loose":
        merge_params = TemporalMergeParams(
            max_cost=32.0,
            min_gap=2.0,
            min_conf=0.30,
            tau_close_disp=2.5,
            conf_improvement_req=0.10,
            max_age_keep=8,
            min_stability_for_strong=2,
            smooth_alpha_strong=0.75,
            smooth_alpha_weak=0.45,
            min_evidence_frames=2,
            weak_conf_scale=0.7,
        )
        default_diff_threshold = 12
    elif args.merge_profile == "calibrated":
        merge_params = TemporalMergeParams(
            max_cost=20.0,
            min_gap=5.0,
            min_conf=0.55,
            tau_close_disp=1.0,
            conf_improvement_req=0.12,
            max_age_keep=14,
            min_stability_for_strong=3,
            smooth_alpha_strong=0.60,
            smooth_alpha_weak=0.30,
            min_evidence_frames=2,
            weak_conf_scale=0.7,
        )
        default_diff_threshold = 12
    else:
        merge_params = TemporalMergeParams()
        default_diff_threshold = 12
    diff_threshold = args.diff_threshold if args.diff_threshold is not None else default_diff_threshold
    run_id = receipts.create_run(
        source=src_path,
        calibration_id=calibration.calibration_id if calibration else None,
        config={
            "width": args.width,
            "height": args.height,
            "every_n": args.every_n,
            "sbs": args.sbs,
            "start_seconds": args.start_seconds,
            "tile_size": args.tile_size,
            "tile_halo": args.tile_halo,
            "merge_profile": args.merge_profile,
            "motion_roi": args.motion_roi,
            "diff_threshold": diff_threshold,
            "debug_loose": args.debug_loose,
            "sweep": args.sweep,
            "max_frames": args.max_frames,
        },
    )
    if dual_file_mode:
        streamer_left = FrameStreamer(
            args.left_file,
            args.width,
            args.height,
            args.every_n,
            gray=True,
            youtube=False,
            start_seconds=args.start_seconds,
        )
        streamer_right = FrameStreamer(
            args.right_file,
            args.width,
            args.height,
            args.every_n,
            gray=True,
            youtube=False,
            start_seconds=args.start_seconds,
        )
        streamer = None
    else:
        streamer = FrameStreamer(
            src_path,
            args.width,
            args.height,
            args.every_n,
            gray=True,
            youtube=bool(args.youtube),
            start_seconds=args.start_seconds,
        )
    print(f"[run] merge profile: {args.merge_profile}")
    print("[run] streaming frames...")
    prev_left = None
    last_seen = None
    disp_state = None
    conf_state = None
    age_state = None
    stability_state = None
    valid_state = None
    evidence_state = None

    try:
        for frame_index in range(args.max_frames):
            if dual_file_mode:
                left, right = streamer_left.get_latest(), streamer_right.get_latest()
                if left is None or right is None:
                    import time

                    t0 = time.time()
                    while time.time() - t0 < args.timeout:
                        left = streamer_left.get_latest()
                        right = streamer_right.get_latest()
                        if left is not None and right is not None:
                            break
                        threading.Event().wait(0.03)
                if left is None or right is None:
                    print(f"[run] failed to grab dual-file frames within {args.timeout}s")
                    sys.exit(1)
            else:
                source_frame, left, right = _wait_for_frame(streamer, args, last_seen)
                if left is None:
                    print(f"[run] failed to grab frame {frame_index} within {args.timeout}s")
                    sys.exit(1)
                last_seen = source_frame

                if args.sbs:
                    mid = left.shape[1] // 2
                    sbs_frame = left
                    left = sbs_frame[:, :mid]
                    right = sbs_frame[:, mid:]

            stage = {}
            import time
            t_start = time.perf_counter()
            if calibration is not None:
                t_rect = time.perf_counter()
                left, right = rectify_pair(left, right, calibration)
                stage["rectify_ms"] = (time.perf_counter() - t_rect) * 1000.0

            t_roi = time.perf_counter()
            roi = build_delta_roi(
                prev_left,
                left,
                diff_threshold=diff_threshold,
                min_luma=int(args.min_frame_mean),
                tile_size=args.tile_size,
                tile_halo=args.tile_halo,
            )
            stage["roi_ms"] = (time.perf_counter() - t_roi) * 1000.0

            sweep_best = None
            t_stereo = time.perf_counter()
            if args.sweep and frame_index == 0:
                try:
                    min_tex_values = _parse_csv_int_list(args.sweep_min_tex, "sweep-min-tex")
                    max_cost_values = _parse_csv_int_list(args.sweep_max_cost, "sweep-max-cost")
                    min_conf_values = _parse_csv_int_list(args.sweep_min_conf, "sweep-min-conf")
                except ValueError as e:
                    print(f"[run] invalid sweep args: {e}")
                    sys.exit(1)
                print("[run] running stereo threshold sweep (start-seconds is fixed per run)")
                sweep_rows = []
                for min_tex in min_tex_values:
                    for max_cost in max_cost_values:
                        for min_conf in min_conf_values:
                            cand, cost_min, conf, valid = stereo_census_roi(
                                left,
                                right,
                                min_tex=min_tex,
                                max_cost=max_cost,
                                min_conf=min_conf,
                                roi_mask=roi["roi_mask"],
                            )
                            valid_count = int(valid.sum())
                            total = int(valid.size)
                            coverage = (100.0 * valid_count / total) if total else 0.0
                            sweep_rows.append(
                                {
                                    "min_tex": min_tex,
                                    "max_cost": max_cost,
                                    "min_conf": min_conf,
                                    "valid_count": valid_count,
                                    "total_pixels": total,
                                    "coverage_pct": f"{coverage:.2f}",
                                }
                            )
                            if sweep_best is None or valid_count > sweep_best["valid_count"]:
                                sweep_best = {
                                    "min_tex": min_tex,
                                    "max_cost": max_cost,
                                    "min_conf": min_conf,
                                    "valid_count": valid_count,
                                    "cand": cand,
                                    "cost": cost_min,
                                    "conf": conf,
                                    "valid": valid,
                                }
                            print(
                                "[sweep] "
                                f"min_tex={min_tex} max_cost={max_cost} min_conf={min_conf} -> "
                                f"{valid_count}/{total} ({coverage:.2f}%)"
                            )
                sweep_path = outdir / "sweep_summary.csv"
                with sweep_path.open("w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["min_tex", "max_cost", "min_conf", "valid_count", "total_pixels", "coverage_pct"],
                    )
                    writer.writeheader()
                    writer.writerows(sweep_rows)
                print(f"[run] wrote sweep summary: {sweep_path}")
                print(f"[run] best sweep config: {sweep_best['min_tex']},{sweep_best['max_cost']},{sweep_best['min_conf']}")
                min_tex = int(sweep_best["min_tex"])
                max_cost = int(sweep_best["max_cost"])
                min_conf = int(sweep_best["min_conf"])
                disp = sweep_best["cand"]
                cost_min = sweep_best["cost"]
                conf = sweep_best["conf"]
                valid = sweep_best["valid"]
            else:
                min_tex = 4 if args.debug_loose else 8
                max_cost = 48 if args.debug_loose else 30
                min_conf = 1 if args.debug_loose else 2
                disp, cost_min, conf, valid = stereo_census_roi(
                    left,
                    right,
                    min_tex=min_tex,
                    max_cost=max_cost,
                    min_conf=min_conf,
                    roi_mask=roi["roi_mask"],
                )
            stage["stereo_ms"] = (time.perf_counter() - t_stereo) * 1000.0

            t_promote = time.perf_counter()
            if disp_state is None:
                h, w = disp.shape
                disp_state = np.zeros((h, w), dtype=np.float32)
                conf_state = np.zeros((h, w), dtype=np.float32)
                age_state = np.zeros((h, w), dtype=np.uint16)
                stability_state = np.zeros((h, w), dtype=np.uint8)
                valid_state = np.zeros((h, w), dtype=bool)
                evidence_state = np.zeros((h, w), dtype=np.uint8)

            cand_disp_px = disp.astype(np.float32) / 256.0
            cand_cost_f = cost_min.astype(np.float32)
            cand_gap_f = conf.astype(np.float32)
            cand_sev = np.zeros_like(cand_gap_f, dtype=np.uint8)
            roi_mask = roi["roi_mask"].astype(np.uint8)

            (
                disp_state,
                conf_state,
                age_state,
                stability_state,
                valid_state,
                evidence_state,
                cand_conf,
                accept_mask,
                merge_stats,
            ) = merge_disparity_state(
                disp_state,
                conf_state,
                age_state,
                stability_state,
                valid_state,
                evidence_state,
                cand_disp_px,
                valid.astype(bool),
                cand_cost_f,
                cand_gap_f,
                cand_sev,
                roi_mask,
                merge_params,
            )
            evidence_state = evidence_state.astype(np.uint8)

            canonical_disp = np.clip(disp_state * 256.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            stage["promote_ms"] = (time.perf_counter() - t_promote) * 1000.0
            stage["frame_ms"] = (time.perf_counter() - t_start) * 1000.0
            depth = None
            if calibration is not None:
                depth = depth_from_disparity(canonical_disp, calibration.q_matrix)

            _save_frame_outputs(
                outdir,
                frame_index,
                left,
                right,
                roi["diff"],
                roi["roi_mask"],
                disp,
                canonical_disp,
                accept_mask.astype(np.uint8),
                depth,
            )
            print_cpu_stats(disp, cost_min, conf, valid)

            frame_id = receipts.write_frame_metrics(
                run_id=run_id,
                frame_index=frame_index,
                stage_metrics=stage,
                roi_tiles=len(roi["tiles"]),
                roi_pixels=int(roi["roi_mask"].sum()),
                roi_coverage=(100.0 * float(roi["roi_mask"].sum()) / float(max(1, roi["roi_mask"].size))),
            )
            accepted_costs = cand_cost_f[accept_mask]
            receipts.write_receipt(
                run_id=run_id,
                frame_id=frame_id,
                kind="disparity",
                roi_set_id=f"frame{frame_index}_tilesz{args.tile_size}",
                decision="promote" if merge_stats["accepted_pixels"] > 0 else "abstain",
                thresholds={
                    "tau_cost": max_cost,
                    "tau_conf": min_conf,
                    "min_tex": min_tex,
                    "tau_close_disp": merge_params.tau_close_disp,
                    "min_evidence_frames": merge_params.min_evidence_frames,
                    "weak_conf_scale": merge_params.weak_conf_scale,
                    "merge_profile": args.merge_profile,
                },
                counts={
                    "promoted": merge_stats["accepted_pixels"],
                    "promoted_close": merge_stats["accepted_close_pixels"],
                    "promoted_better": merge_stats["accepted_better_pixels"],
                    "promoted_temporal": merge_stats["accepted_temporal_pixels"],
                    "expired": merge_stats["expired_pixels"],
                    "roi_pixels": int(roi["roi_mask"].sum()),
                },
                residual_mean=float(accepted_costs.mean()) if accepted_costs.size else 0.0,
                residual_p95=float(np.percentile(accepted_costs, 95)) if accepted_costs.size else 0.0,
                invariants={
                    "valid_mask_nonzero": bool(valid.any()),
                    "roi_nonempty": bool(roi["roi_mask"].any()),
                    "calibrated": calibration is not None,
                },
            )
            for kind, name in [
                ("left_input", f"left_input_f{frame_index:04d}.png"),
                ("right_input", f"right_input_f{frame_index:04d}.png"),
                ("roi_mask", f"roi_mask_f{frame_index:04d}.png"),
                ("candidate_disp", f"candidate_disp_f{frame_index:04d}.png"),
                ("canonical_disp", f"canonical_disp_f{frame_index:04d}.png"),
            ]:
                receipts.write_artifact(run_id, frame_id, kind, outdir / name)
            if depth is not None:
                receipts.write_artifact(run_id, frame_id, "depth", outdir / f"depth_f{frame_index:04d}.png")

            print(
                "[run] frame "
                f"{frame_index}: tiles={len(roi['tiles'])} roi_px={int(roi['roi_mask'].sum())} "
                f"promoted={merge_stats['accepted_pixels']} expired={merge_stats['expired_pixels']}"
            )
            prev_left = left.copy()

        print(f"[run] saved outputs to {outdir}")
        print(f"[run] wrote receipts to {args.receipts_db}")

        if args.gpu and HAS_VULKAN:
            try:
                run_vulkan(left, right)
            except Exception as e:
                import traceback
                print(f"[run][gpu] failed: {e}")
                traceback.print_exc()
        elif args.gpu:
            print("[run][gpu] Vulkan not available; skipping.")
    finally:
        receipts.close()
        if dual_file_mode:
            streamer_left.close()
            streamer_right.close()
        else:
            streamer.close()


if __name__ == "__main__":
    main()
