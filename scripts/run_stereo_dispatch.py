#!/usr/bin/env python3
"""
End-to-end stereo runtime:
- streams frames from a YouTube URL or local file via ffmpeg (no full download)
- optional side-by-side split for 3D videos
- can load a fixed-rig calibration artifact and rectify each pair
- builds motion-gated delta ROI tiles by default, with `--full-frame-roi` as an explicit override
  for static scenes; computes CPU census stereo candidates, promotes canonical disparity, and
  writes append-only receipts to SQLite
- if Vulkan is available, prints pipeline readiness (reuses vk_stereo_pipeline bindings);
  GPU dispatch is left as a TODO hook.

Usage examples:
  python scripts/run_stereo_dispatch.py --youtube https://youtu.be/... --width 640 --height 360 --every-n 2 --sbs
  python scripts/run_stereo_dispatch.py --file /path/to/video.mp4 --width 640 --height 360
"""

import argparse
import atexit
import csv
import re
import struct
import subprocess
import threading
import collections
import json
import time
import zlib
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
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
from self_calibrate_stereo import estimate_self_calibration_from_arrays

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
    def __init__(
        self,
        src: str,
        width: int,
        height: int,
        every_n: int,
        gray: bool,
        youtube: bool,
        start_seconds: float = 0.0,
        auto_res: bool = False,
        status_prefix: str = "[stream]",
    ):
        self.gray = gray
        self.status_prefix = status_prefix
        self._closing = False
        self._saw_broken_pipe = False
        pix_fmt = "gray" if gray else "rgb24"
        print(f"{self.status_prefix} initializing frame streamer for {src}")
        input_arg = src if not youtube else self._youtube_best(src)
        if auto_res:
            print(f"{self.status_prefix} probing source resolution for auto-res")
            probed_width, probed_height = self._probe_source_size(input_arg)
            if probed_width is not None and probed_height is not None:
                width, height = probed_width, probed_height
                print(f"{self.status_prefix} auto-resolved source size: {width}x{height}")
            else:
                print(f"{self.status_prefix} auto-res probe failed; falling back to requested width/height")
        self.width = width
        self.height = height
        # `showinfo` emits selected-frame indices and pts_time on stderr so separate
        # oracle/runtime runs can be aligned by source time instead of loop ordinal.
        vf = f"select='not(mod(n\\,{every_n}))',showinfo,scale={width}:{height},format={pix_fmt}"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "info",
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
        self.meta_buf = collections.deque(maxlen=120)
        self.packet_counter = 0
        print(f"{self.status_prefix} starting ffmpeg reader at {width}x{height}")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 ** 6,
        )
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        self.stderr_thread = threading.Thread(target=self._stderr_reader, daemon=True)
        self.stderr_thread.start()
        atexit.register(self.close)

    def _youtube_best(self, url: str) -> str:
        import yt_dlp

        print(f"{self.status_prefix} resolving youtube stream URL")
        t0 = time.perf_counter()
        ydl = yt_dlp.YoutubeDL(
            {
                "quiet": True,
                "format": "bestvideo[ext=mp4][vcodec~='(avc1|h264)']/best",
                "extractor_args": {"youtube": {"player_client": ["android"]}},
                "socket_timeout": 15,
            }
        )
        info = ydl.extract_info(url, download=False)
        print(f"{self.status_prefix} resolved youtube stream URL in {time.perf_counter() - t0:.1f}s")
        return info["url"]

    def _probe_source_size(self, input_arg: str):
        print(f"{self.status_prefix} probing source size via ffprobe")
        t0 = time.perf_counter()
        try:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "json",
                    input_arg,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            print(f"{self.status_prefix} ffprobe size probe timed out in {time.perf_counter() - t0:.1f}s")
            return None, None
        if probe.returncode != 0 or not probe.stdout.strip():
            print(f"{self.status_prefix} ffprobe size probe failed in {time.perf_counter() - t0:.1f}s")
            return None, None
        try:
            payload = json.loads(probe.stdout)
        except json.JSONDecodeError:
            print(f"{self.status_prefix} ffprobe returned invalid JSON in {time.perf_counter() - t0:.1f}s")
            return None, None
        for stream in payload.get("streams", []):
            width = stream.get("width")
            height = stream.get("height")
            if width and height:
                print(f"{self.status_prefix} ffprobe size probe completed in {time.perf_counter() - t0:.1f}s")
                return int(width), int(height)
        print(f"{self.status_prefix} ffprobe size probe found no dimensions in {time.perf_counter() - t0:.1f}s")
        return None, None

    def _reader(self):
        count = 0
        while True:
            stdout = self.proc.stdout
            if stdout is None:
                break
            try:
                raw = stdout.read(self.frame_bytes)
            except (ValueError, OSError):
                if self._closing:
                    break
                print(f"{self.status_prefix} ffmpeg stdout closed unexpectedly")
                break
            if len(raw) != self.frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8)
            if self.gray:
                frame = frame.reshape(self.height, self.width)
            else:
                frame = frame.reshape(self.height, self.width, 3)
            meta = self.meta_buf.popleft() if self.meta_buf else {}
            packet = {
                "frame": frame,
                "selected_index": int(meta.get("selected_index", self.packet_counter)),
                "pts_time": meta.get("pts_time"),
            }
            self.buf.append(packet)
            self.packet_counter += 1
            count += 1
        print(f"{self.status_prefix} frame reader stopped after {count} frames")

    def _stderr_reader(self):
        if self.proc.stderr is None:
            return
        showinfo_re = re.compile(r"n:\s*(\d+).*?pts_time:\s*([+\-0-9.eE]+)")
        while True:
            raw = self.proc.stderr.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if "Parsed_showinfo" in line:
                m = showinfo_re.search(line)
                if m:
                    self.meta_buf.append(
                        {
                            "selected_index": int(m.group(1)),
                            "pts_time": float(m.group(2)),
                        }
                    )
                continue
            if line.startswith("Last message repeated"):
                continue
            if "Broken pipe" in line or "Error closing file: Broken pipe" in line:
                self._saw_broken_pipe = True
                continue
            if self._closing and ("Error writing trailer" in line or "Error muxing a packet" in line):
                continue
            print(f"{self.status_prefix} ffmpeg: {line}")
        if self._saw_broken_pipe and self._closing:
            print(f"{self.status_prefix} ffmpeg writer closed after consumer exit")

    def get_pair(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if len(self.buf) < 2:
            return None, None
        return self.buf[-2]["frame"], self.buf[-1]["frame"]

    def get_latest(self) -> Optional[np.ndarray]:
        if not self.buf:
            return None
        return self.buf[-1]["frame"]

    def get_latest_packet(self):
        if not self.buf:
            return None
        return self.buf[-1]

    def close(self):
        if getattr(self, "_closing", False):
            return
        self._closing = True
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                print(f"{self.status_prefix} ffmpeg did not exit on terminate; killing")
                proc.kill()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    print(f"{self.status_prefix} ffmpeg process still alive after kill")
        try:
            if proc.stderr is not None:
                proc.stderr.close()
        except Exception:
            pass


def _packet_token(packet):
    if packet is None:
        return None
    pts_time = packet.get("pts_time")
    if pts_time is not None:
        return (packet.get("selected_index"), round(float(pts_time), 6))
    return (packet.get("selected_index"), None)


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


def stereo_sgbm_roi(
    left: np.ndarray,
    right: np.ndarray,
    d_min=0,
    d_max=128,
    block_size=7,
    uniqueness_ratio=10,
    roi_mask: Optional[np.ndarray] = None,
    expand_texture_min: int = 24,
    expand_lr_max: int = 48,
    expand_median_max: int = 32,
):
    import cv2
    import numpy as np

    num_disparities = max(16, ((int(d_max - d_min + 1) + 15) // 16) * 16)
    block_size = max(5, int(block_size) | 1)
    stereo_left = cv2.StereoSGBM_create(
        minDisparity=int(d_min),
        numDisparities=int(num_disparities),
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=int(uniqueness_ratio),
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM,
    )
    stereo_right = cv2.StereoSGBM_create(
        minDisparity=int(d_min),
        numDisparities=int(num_disparities),
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=int(uniqueness_ratio),
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM,
    )
    disp_left_q4 = stereo_left.compute(left, right).astype(np.int16)
    disp_right_q4 = stereo_right.compute(right, left).astype(np.int16)

    # A second half-resolution pass cheaply recovers broader low-texture
    # regions that full-resolution SGBM tends to miss. The coarse disparity is
    # only used to rescue invalid full-res pixels; it does not replace local
    # full-res matches where they already exist.
    left_half = cv2.pyrDown(left)
    right_half = cv2.pyrDown(right)
    d_min_half = int(np.floor(d_min / 2.0))
    d_max_half = max(d_min_half + 16, int(np.ceil(d_max / 2.0)))
    num_disparities_half = max(16, ((int(d_max_half - d_min_half + 1) + 15) // 16) * 16)
    stereo_left_half = cv2.StereoSGBM_create(
        minDisparity=int(d_min_half),
        numDisparities=int(num_disparities_half),
        blockSize=max(5, block_size + 2),
        P1=8 * (block_size + 2) * (block_size + 2),
        P2=32 * (block_size + 2) * (block_size + 2),
        disp12MaxDiff=1,
        uniquenessRatio=max(5, int(uniqueness_ratio) - 2),
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM,
    )
    stereo_right_half = cv2.StereoSGBM_create(
        minDisparity=int(d_min_half),
        numDisparities=int(num_disparities_half),
        blockSize=max(5, block_size + 2),
        P1=8 * (block_size + 2) * (block_size + 2),
        P2=32 * (block_size + 2) * (block_size + 2),
        disp12MaxDiff=1,
        uniquenessRatio=max(5, int(uniqueness_ratio) - 2),
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM,
    )
    disp_left_half_q4 = stereo_left_half.compute(left_half, right_half).astype(np.int16)
    disp_right_half_q4 = stereo_right_half.compute(right_half, left_half).astype(np.int16)
    disp_left_half_up_q4 = cv2.resize(
        (disp_left_half_q4.astype(np.float32) * 2.0),
        (left.shape[1], left.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.int16)
    disp_right_half_up_q4 = cv2.resize(
        (disp_right_half_q4.astype(np.float32) * 2.0),
        (right.shape[1], right.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.int16)

    rescue_mask = (disp_left_q4 <= 0) & (disp_left_half_up_q4 > 0)
    if roi_mask is not None:
        rescue_mask &= roi_mask != 0
    if np.any(rescue_mask):
        disp_left_q4 = disp_left_q4.copy()
        disp_right_q4 = disp_right_q4.copy()
        disp_left_q4[rescue_mask] = disp_left_half_up_q4[rescue_mask]
        disp_right_q4[rescue_mask] = disp_right_half_up_q4[rescue_mask]

    base_valid = disp_left_q4 > 0
    valid = base_valid.astype(np.uint8)
    if roi_mask is not None:
        valid = (valid & (roi_mask != 0).astype(np.uint8)).astype(np.uint8)
    disp_q8 = np.zeros_like(disp_left_q4, dtype=np.uint16)
    disp_q8[base_valid] = (disp_left_q4[base_valid].astype(np.uint16) << 4)
    if roi_mask is not None:
        disp_q8[roi_mask == 0] = 0

    cost_min = np.full(left.shape, 255, dtype=np.uint16)
    conf = np.zeros(left.shape, dtype=np.uint16)
    lr_delta_map = np.full(left.shape, 255, dtype=np.uint16)
    median_delta_map = np.full(left.shape, 255, dtype=np.uint16)
    texture_map = np.zeros(left.shape, dtype=np.uint16)
    disp_grad_map = np.full(left.shape, 255, dtype=np.uint16)
    edge_distance_map = np.zeros(left.shape, dtype=np.uint16)
    border_penalty_map = np.full(left.shape, 255, dtype=np.uint16)

    grad_x = cv2.Sobel(left, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(left, cv2.CV_16S, 0, 1, ksize=3)
    texture = np.abs(grad_x).astype(np.int32) + np.abs(grad_y).astype(np.int32)
    texture_score = np.clip(texture // 12, 0, 15).astype(np.uint16)
    disp_median_q4 = cv2.medianBlur(disp_left_q4.astype(np.int16), 5)
    disp_grad_x = cv2.Sobel(disp_left_q4.astype(np.int16), cv2.CV_16S, 1, 0, ksize=3)
    disp_grad_y = cv2.Sobel(disp_left_q4.astype(np.int16), cv2.CV_16S, 0, 1, ksize=3)
    disp_grad = np.abs(disp_grad_x).astype(np.int32) + np.abs(disp_grad_y).astype(np.int32)

    h, w = left.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    edge_dist = np.minimum.reduce([xx, yy, (w - 1) - xx, (h - 1) - yy])
    edge_dist_norm = np.clip(edge_dist / 32.0, 0.0, 1.0)
    border_penalty = 1.0 - edge_dist_norm
    edge_distance_u8 = np.clip(edge_dist_norm * 255.0, 0, 255).astype(np.uint16)
    border_penalty_u8 = np.clip(border_penalty * 255.0, 0, 255).astype(np.uint16)

    ys, xs = np.nonzero(valid)
    for y, x in zip(ys.tolist(), xs.tolist()):
        d_q4 = int(disp_left_q4[y, x])
        xr = x - (d_q4 // 16)
        if xr < 0 or xr >= right.shape[1]:
            continue
        dr_q4 = int(disp_right_q4[y, xr])
        lr_delta = abs(d_q4 - dr_q4)
        median_delta = abs(d_q4 - int(disp_median_q4[y, x]))
        lr_delta_u8 = int(min(255, lr_delta // 8))
        median_delta_u8 = int(min(255, median_delta // 8))
        disp_grad_local = int(min(255, disp_grad[y, x] // 16))
        texture_local = int(texture_score[y, x])
        edge_norm = float(edge_dist_norm[y, x])
        border = float(border_penalty[y, x])
        interior_bonus = 1.0 if (edge_norm >= 0.35 and texture_local >= 6 and disp_grad_local >= 8) else 0.0

        lr_cost = min(40, lr_delta_u8)
        median_cost = min(20, median_delta_u8)
        border_cost = int(round(6.0 * border))
        texture_credit = 3 if interior_bonus > 0.0 else 0
        edge_relief = 2 if edge_norm >= 0.35 else 0
        cost_min[y, x] = max(0, lr_cost + median_cost + border_cost - texture_credit - edge_relief)

        lr_score = int(round(12.0 * float(np.exp(-lr_delta_u8 / 24.0))))
        median_score = int(round(8.0 * float(np.exp(-median_delta_u8 / 18.0))))
        edge_score = int(round(edge_norm * 4.0))
        bonus_score = 3 if interior_bonus > 0.0 else 0
        border_penalty_score = int(round(border * 3.0))
        grad_weight = min(1.0, disp_grad_local / 24.0)
        grad_bonus = int(round(4.0 * grad_weight))
        conf_val = texture_local + lr_score + median_score + edge_score + grad_bonus + bonus_score - border_penalty_score
        conf[y, x] = int(min(31, max(0, conf_val)))
        lr_delta_map[y, x] = lr_delta_u8
        median_delta_map[y, x] = median_delta_u8
        texture_map[y, x] = min(255, texture_local * 8)
        disp_grad_map[y, x] = disp_grad_local
        edge_distance_map[y, x] = edge_distance_u8[y, x]
        border_penalty_map[y, x] = border_penalty_u8[y, x]

    evidence = {
        "lr_delta": lr_delta_map,
        "median_delta": median_delta_map,
        "texture": texture_map,
        "disp_gradient": disp_grad_map,
        "edge_distance": edge_distance_map,
        "border_penalty": border_penalty_map,
    }

    # Recover thin missed structures cheaply before region filtering. Use a
    # small isotropic pass everywhere plus a slightly wider interior-biased
    # horizontal pass to target the recurring center/right FN bands.
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel_h = np.ones((3, 5), dtype=np.uint8)
    valid_u8 = valid.astype(np.uint8)
    grown = cv2.dilate(valid_u8, kernel, iterations=1)
    x_norm = xx / max(float(w - 1), 1.0)
    interior_band = (x_norm >= 0.38) & (x_norm <= 0.82) & (edge_dist_norm >= 0.30)
    grown_h = cv2.dilate(valid_u8, kernel_h, iterations=1)
    expansion = grown & (~valid_u8.astype(bool))
    expansion_h = grown_h & (~valid_u8.astype(bool))
    support = np.ones_like(valid_u8, dtype=bool)
    if roi_mask is not None:
        support &= roi_mask != 0
    support &= texture_map >= int(expand_texture_min)
    support &= lr_delta_map <= int(expand_lr_max)
    support &= median_delta_map <= int(expand_median_max)
    expansion &= support

    support_h = support.copy()
    support_h &= interior_band
    support_h &= texture_map >= max(int(expand_texture_min), 28)
    support_h &= disp_grad_map >= 8
    expansion_h &= support_h

    expansion_u8 = (expansion | expansion_h).astype(np.uint8)

    if np.any(expansion_u8):
        disp_q8_d = cv2.dilate(disp_q8, kernel, iterations=1)
        conf_d = cv2.dilate(conf, kernel, iterations=1)
        texture_d = cv2.dilate(texture_map, kernel, iterations=1)
        disp_grad_d = cv2.dilate(disp_grad_map, kernel, iterations=1)
        edge_distance_d = cv2.dilate(edge_distance_map, kernel, iterations=1)
        border_penalty_e = cv2.erode(border_penalty_map, kernel, iterations=1)
        cost_min_e = cv2.erode(cost_min, kernel, iterations=1)
        lr_delta_e = cv2.erode(lr_delta_map, kernel, iterations=1)
        median_delta_e = cv2.erode(median_delta_map, kernel, iterations=1)

        disp_q8[expansion] = disp_q8_d[expansion]
        conf[expansion] = conf_d[expansion]
        texture_map[expansion] = texture_d[expansion]
        disp_grad_map[expansion] = disp_grad_d[expansion]
        edge_distance_map[expansion] = edge_distance_d[expansion]
        border_penalty_map[expansion] = border_penalty_e[expansion]
        cost_min[expansion] = cost_min_e[expansion]
        lr_delta_map[expansion] = lr_delta_e[expansion]
        median_delta_map[expansion] = median_delta_e[expansion]
        valid = (valid_u8 | expansion_u8).astype(np.uint8)

    return disp_q8, cost_min, conf, valid, evidence


def motion_roi_mask(prev_frame: np.ndarray, curr_frame: np.ndarray, diff_threshold: int = 12, min_luma: int = 5):
    diff = np.abs(curr_frame.astype(np.int16) - prev_frame.astype(np.int16)).astype(np.uint16)
    bright = np.maximum(prev_frame, curr_frame) >= min_luma
    mask = ((diff >= diff_threshold) & bright).astype(np.uint8)
    return diff, mask


def full_frame_roi(frame: np.ndarray, tile_size: int):
    """Return a full-frame ROI when motion gating is not desired."""
    h, w = frame.shape
    roi_mask = np.ones((h, w), dtype=np.uint8)
    tile_mask = np.ones(((h + tile_size - 1) // tile_size, (w + tile_size - 1) // tile_size), dtype=np.uint8)
    tiles = []
    for ty in range(tile_mask.shape[0]):
        y0 = ty * tile_size
        y1 = min(y0 + tile_size, h)
        for tx in range(tile_mask.shape[1]):
            x0 = tx * tile_size
            x1 = min(x0 + tile_size, w)
            tiles.append((x0, y0, x1 - x0, y1 - y0))
    return {
        "diff": np.zeros((h, w), dtype=np.uint16),
        "pixel_mask": roi_mask.copy(),
        "tile_mask": tile_mask,
        "roi_mask": roi_mask,
        "tiles": tiles,
    }


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _encode_png(arr: np.ndarray) -> bytes:
    if arr.ndim == 2:
        color_type = 0  # grayscale
        channels = 1
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        channels = arr.shape[2]
        color_type = 2 if channels == 3 else 6
    else:
        raise ValueError(f"unsupported PNG array shape: {arr.shape}")

    if arr.dtype != np.uint8:
        raise TypeError(f"PNG encoder requires uint8 input, got {arr.dtype}")

    height, width = arr.shape[:2]
    if channels == 1:
        raw = arr
    else:
        raw = arr.reshape(height, width * channels)

    # PNG scanlines are prefixed with a filter byte. Use filter 0 for simplicity.
    scanlines = b"".join(b"\x00" + row.tobytes() for row in raw)
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + _png_chunk(b"IEND", b"")
    )


def save_png(arr: np.ndarray, path: Path, scale: float = 1.0):
    if arr.dtype != np.uint8:
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_encode_png(arr))


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_confidence_model(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _apply_confidence_model(
    model: dict,
    *,
    valid: np.ndarray,
    roi_mask: np.ndarray,
    disp_q8: np.ndarray,
    cost_u8: np.ndarray,
    heuristic_conf_u8: np.ndarray,
    lr_delta_u8: np.ndarray,
    median_delta_u8: np.ndarray,
    texture_u8: np.ndarray,
    disp_gradient_u8: np.ndarray,
    edge_distance_u8: np.ndarray,
    border_penalty_u8: np.ndarray,
) -> np.ndarray:
    weights = model["weights"]
    mean = model["feature_mean"]
    std = model["feature_std"]

    features = {
        "runtime_roi": (roi_mask != 0).astype(np.float32),
        "runtime_candidate": (valid != 0).astype(np.float32),
        "runtime_promoted": np.zeros_like(valid, dtype=np.float32),
        "candidate_disp_u8": disp_q8.astype(np.float32) / 255.0,
        "candidate_cost_u8": cost_u8.astype(np.float32) / 255.0,
        "candidate_conf_u8": heuristic_conf_u8.astype(np.float32) / 255.0,
        "candidate_lr_delta_u8": lr_delta_u8.astype(np.float32) / 255.0,
        "candidate_median_delta_u8": median_delta_u8.astype(np.float32) / 255.0,
        "candidate_texture_u8": texture_u8.astype(np.float32) / 255.0,
        "candidate_disp_gradient_u8": disp_gradient_u8.astype(np.float32) / 255.0,
        "candidate_edge_distance_u8": edge_distance_u8.astype(np.float32) / 255.0,
        "candidate_border_penalty_u8": border_penalty_u8.astype(np.float32) / 255.0,
    }

    score = np.full(valid.shape, float(weights["bias"]), dtype=np.float32)
    for name, arr in features.items():
        centered = (arr - float(mean[name])) / max(1e-6, float(std[name]))
        score += float(weights[name]) * centered
    score = np.clip(score, -40.0, 40.0)
    prob = 1.0 / (1.0 + np.exp(-score))
    prob[valid == 0] = 0.0
    return prob.astype(np.float32)


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


def _wait_for_frame(streamer: FrameStreamer, args, last_seen, status_prefix: str = "[run]", heartbeat_s: float = 3.0):
    import time

    t0 = time.time()
    last_report = t0
    while time.time() - t0 < args.timeout:
        latest_packet = streamer.get_latest_packet()
        latest = None if latest_packet is None else latest_packet["frame"]
        latest_token = _packet_token(latest_packet)
        if latest is None or latest_token == last_seen:
            now = time.time()
            if now - last_report >= heartbeat_s:
                print(f"{status_prefix} waiting for first frame... {now - t0:.1f}s elapsed")
                last_report = now
            threading.Event().wait(0.03)
            continue
        if float(latest.mean()) < args.min_frame_mean:
            now = time.time()
            if now - last_report >= heartbeat_s:
                print(f"{status_prefix} waiting for usable frame brightness... {now - t0:.1f}s elapsed")
                last_report = now
            threading.Event().wait(0.03)
            continue
        if args.sbs:
            return latest_packet, latest, latest
        left, right = streamer.get_pair()
        if left is None or right is None:
            now = time.time()
            if now - last_report >= heartbeat_s:
                print(f"{status_prefix} waiting for stereo pair... {now - t0:.1f}s elapsed")
                last_report = now
            threading.Event().wait(0.03)
            continue
        return latest_packet, left, right
    return None


def _seed_and_self_calibrate(
    *,
    dual_file_mode: bool,
    left_streamer: Optional[FrameStreamer],
    right_streamer: Optional[FrameStreamer],
    streamer: Optional[FrameStreamer],
    args,
    timeout: float,
    selfcal_output: Path,
    outdir: Path,
    rig_id: str = "runtime_selfcal",
    max_features: int = 4000,
    save_debug: bool = False,
    status_prefix: str = "[run]",
):
    seed = None
    if dual_file_mode:
        if left_streamer is None or right_streamer is None:
            return None, None
        t0 = time.time()
        last_report = t0
        left = None
        right = None
        print(f"{status_prefix} waiting for dual-file seed frames...")
        while time.time() - t0 < timeout:
            left = left_streamer.get_latest()
            right = right_streamer.get_latest()
            if left is not None and right is not None:
                break
            now = time.time()
            if now - last_report >= 5.0:
                print(f"{status_prefix} waiting for dual-file seed frames... {now - t0:.1f}s elapsed")
                last_report = now
            threading.Event().wait(0.03)
        if left is None or right is None:
            return None, None
        source_frame = None
        seed = (source_frame, left, right)
    else:
        if streamer is None:
            return None, None
        seed_frame = _wait_for_frame(streamer, args, last_seen=None)
        if seed_frame is None:
            return None, None
        source_frame, left, right = seed_frame
        if left is None or right is None:
            return None, None
        seed = (source_frame, left, right)

    try:
        print(f"{status_prefix} bootstrapping self-calibration from first stereo pair...")
        t_boot = time.perf_counter()
        _, left_img, right_img = seed
        if args.sbs:
            sbs_frame = left_img
            mid = sbs_frame.shape[1] // 2
            left_img = sbs_frame[:, :mid]
            right_img = sbs_frame[:, mid:]
        cal = estimate_self_calibration_from_arrays(
            left_img,
            right_img,
            rig_id=rig_id,
            max_features=max_features,
        )
        np.savez_compressed(
            selfcal_output,
            left_map_x=cal["left_map_x"],
            left_map_y=cal["left_map_y"],
            right_map_x=cal["right_map_x"],
            right_map_y=cal["right_map_y"],
            H1=cal["H1"],
            H2=cal["H2"],
            F=cal["F"],
            Q=cal["Q"],
        )
        (selfcal_output.with_suffix(".json")).write_text(json.dumps(cal["metadata"], sort_keys=True, indent=2))
        if save_debug:
            import cv2

            debug_dir = outdir / "selfcal_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            rect_left = cv2.remap(
                left_img,
                cal["left_map_x"],
                cal["left_map_y"],
                interpolation=cv2.INTER_LINEAR,
            )
            rect_right = cv2.remap(
                right_img,
                cal["right_map_x"],
                cal["right_map_y"],
                interpolation=cv2.INTER_LINEAR,
            )
            cv2.imwrite(str(debug_dir / "selfcal_left.png"), rect_left)
            cv2.imwrite(str(debug_dir / "selfcal_right.png"), rect_right)
        cal = load_calibration_artifact(selfcal_output)
        print(f"{status_prefix} self-calibration bootstrap completed in {time.perf_counter() - t_boot:.1f}s")
        return seed, cal
    except Exception as exc:
        return seed, exc


def _save_frame_outputs(
    outdir: Path,
    frame_index: int,
    left: np.ndarray,
    right: np.ndarray,
    diff: np.ndarray,
    roi_mask: np.ndarray,
    candidate: np.ndarray,
    candidate_valid: np.ndarray,
    candidate_cost: np.ndarray,
    candidate_conf: np.ndarray,
    candidate_lr_delta: np.ndarray,
    candidate_median_delta: np.ndarray,
    candidate_texture: np.ndarray,
    candidate_disp_gradient: np.ndarray,
    candidate_edge_distance: np.ndarray,
    candidate_border_penalty: np.ndarray,
    canonical: np.ndarray,
    promoted_mask: np.ndarray,
    depth: Optional[np.ndarray],
):
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
    save_png(candidate_valid * 255, outdir / f"candidate_mask_f{frame_index:04d}.png")
    save_png(np.clip(candidate_cost, 0, 255).astype(np.uint8), outdir / f"candidate_cost_f{frame_index:04d}.png")
    save_png(np.clip(candidate_conf, 0, 255).astype(np.uint8), outdir / f"candidate_conf_f{frame_index:04d}.png")
    save_png(np.clip(candidate_lr_delta, 0, 255).astype(np.uint8), outdir / f"candidate_lr_delta_f{frame_index:04d}.png")
    save_png(np.clip(candidate_median_delta, 0, 255).astype(np.uint8), outdir / f"candidate_median_delta_f{frame_index:04d}.png")
    save_png(np.clip(candidate_texture, 0, 255).astype(np.uint8), outdir / f"candidate_texture_f{frame_index:04d}.png")
    save_png(np.clip(candidate_disp_gradient, 0, 255).astype(np.uint8), outdir / f"candidate_disp_gradient_f{frame_index:04d}.png")
    save_png(np.clip(candidate_edge_distance, 0, 255).astype(np.uint8), outdir / f"candidate_edge_distance_f{frame_index:04d}.png")
    save_png(np.clip(candidate_border_penalty, 0, 255).astype(np.uint8), outdir / f"candidate_border_penalty_f{frame_index:04d}.png")
    save_png((canonical >> 8).astype(np.uint8), outdir / f"canonical_disp_f{frame_index:04d}.png", scale=4)
    save_png(promoted_mask * 255, outdir / f"promoted_mask_f{frame_index:04d}.png")
    save_npz(
        outdir / f"promoted_depth_f{frame_index:04d}.npz",
        canonical_disp_q8=canonical.astype(np.uint16),
        promoted_mask=promoted_mask.astype(np.uint8),
        depth=(depth.astype(np.float32) if depth is not None else np.zeros_like(canonical, dtype=np.float32)),
    )
    if depth_vis is not None:
        save_png(depth_vis, outdir / f"depth_f{frame_index:04d}.png", scale=1.0)
    save_png(left.astype(np.uint8), outdir / "left_input.png")
    save_png(right.astype(np.uint8), outdir / "right_input.png")
    save_png(roi_mask * 255, outdir / "roi_mask.png")
    save_png((candidate >> 8).astype(np.uint8), outdir / "disp_cpu.png", scale=4)
    save_png(candidate_valid * 255, outdir / "candidate_mask.png")
    save_png(np.clip(candidate_cost, 0, 255).astype(np.uint8), outdir / "candidate_cost.png")
    save_png(np.clip(candidate_conf, 0, 255).astype(np.uint8), outdir / "candidate_conf.png")
    save_png(np.clip(candidate_lr_delta, 0, 255).astype(np.uint8), outdir / "candidate_lr_delta.png")
    save_png(np.clip(candidate_median_delta, 0, 255).astype(np.uint8), outdir / "candidate_median_delta.png")
    save_png(np.clip(candidate_texture, 0, 255).astype(np.uint8), outdir / "candidate_texture.png")
    save_png(np.clip(candidate_disp_gradient, 0, 255).astype(np.uint8), outdir / "candidate_disp_gradient.png")
    save_png(np.clip(candidate_edge_distance, 0, 255).astype(np.uint8), outdir / "candidate_edge_distance.png")
    save_png(np.clip(candidate_border_penalty, 0, 255).astype(np.uint8), outdir / "candidate_border_penalty.png")
    save_png((canonical >> 8).astype(np.uint8), outdir / "canonical_disp.png", scale=4)
    save_png(promoted_mask * 255, outdir / "valid_cpu.png")
    save_npz(
        outdir / "promoted_depth.npz",
        canonical_disp_q8=canonical.astype(np.uint16),
        promoted_mask=promoted_mask.astype(np.uint8),
        depth=(depth.astype(np.float32) if depth is not None else np.zeros_like(canonical, dtype=np.float32)),
    )
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
    ap.add_argument("--auto-res", action="store_true", help="Probe the source and scale to the native decoded resolution")
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
    ap.add_argument("--region-min-pixels", type=int, help="Minimum connected accepted pixels required for region-level promotion")
    ap.add_argument("--region-max-disp-std", type=float, help="Maximum disparity std-dev allowed inside an accepted region")
    ap.add_argument("--region-min-fill-ratio", type=float, help="Minimum accepted-pixel fill ratio inside a region bounding box")
    ap.add_argument(
        "--merge-profile",
        choices=["demo", "demo_loose", "calibrated", "calibrated_sgbm", "strict"],
        default="strict",
        help="Temporal merge preset",
    )
    ap.add_argument("--gpu", action="store_true", help="Attempt GPU dispatch (Vulkan) after CPU")
    ap.add_argument("--min-frame-mean", type=float, default=5.0, help="Wait until frame mean exceeds this before locking input")
    ap.add_argument("--sweep", action="store_true", help="Evaluate stereo matcher threshold combinations and report summary")
    ap.add_argument("--sweep-min-tex", default="4,8", help="Comma-separated min_tex values for sweep")
    ap.add_argument("--sweep-max-cost", default="24,30,48", help="Comma-separated max_cost values for sweep")
    ap.add_argument("--sweep-min-conf", default="0,1,2", help="Comma-separated min_conf values for sweep")
    ap.add_argument(
        "--matcher",
        choices=["auto", "cpu_census", "opencv_sgbm"],
        default="auto",
        help="Candidate disparity generator to use in the runtime",
    )
    ap.add_argument("--disp-min", type=int, default=0, help="Minimum disparity for the CPU census matcher")
    ap.add_argument("--disp-max", type=int, help="Maximum disparity for the CPU census matcher")
    ap.add_argument("--sgbm-block-size", type=int, default=7, help="OpenCV SGBM block size for calibrated matcher mode")
    ap.add_argument("--sgbm-uniqueness", type=int, default=10, help="OpenCV SGBM uniqueness ratio for calibrated matcher mode")
    ap.add_argument("--expand-texture-min", type=int, default=24, help="Minimum texture map value required for cheap SGBM candidate expansion")
    ap.add_argument("--expand-lr-max", type=int, default=48, help="Maximum LR delta allowed for cheap SGBM candidate expansion")
    ap.add_argument("--expand-median-max", type=int, default=32, help="Maximum median-delta allowed for cheap SGBM candidate expansion")
    ap.add_argument("--confidence-model", type=str, help="Optional learned confidence model JSON for calibrated SGBM runs")
    ap.add_argument("--region-model", type=str, help="Optional learned region model JSON for calibrated region-aware promotion")
    ap.add_argument("--region-score-threshold-strong", type=float, help="Strong acceptance threshold for learned region scoring")
    ap.add_argument("--region-score-threshold-weak", type=float, help="Weak acceptance threshold for learned region scoring")
    ap.add_argument("--debug-loose", action="store_true", help="Relax matcher thresholds for coverage/debugging")
    ap.add_argument("--motion-roi", action="store_true", help="Explicitly use the default frame-diff ROI gating path")
    ap.add_argument("--full-frame-roi", action="store_true", help="Force full-frame stereo search instead of frame-diff ROI gating")
    ap.add_argument("--diff-threshold", type=int, default=None, help="Frame diff threshold for motion ROI (defaults vary by profile)")
    ap.add_argument("--selfcal-max-features", type=int, default=4000, help="ORB feature budget for bootstrap self-calibration")
    ap.add_argument("--force-unrectified", action="store_true", help="Skip self-calibration bootstrap and force unrectified demo mode")
    ap.add_argument("--selfcal-rig-id", type=str, default="runtime_selfcal", help="Rig ID for generated self-calibration artifact")
    ap.add_argument("--selfcal-debug", action="store_true", help="Write rectified self-calibration preview images")
    args = ap.parse_args()
    if not (args.youtube or args.file or (args.left_file and args.right_file)):
        ap.error("provide --youtube, --file, or both --left-file and --right-file")
    if args.full_frame_roi and args.motion_roi:
        print("[run] --full-frame-roi overrides --motion-roi; running full-frame search")

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
    selfcal_output = outdir / "selfcal_bootstrap.npz"
    bootstrap_seed = None
    bootstrap_error = None

    if args.force_unrectified:
        print("[run] --force-unrectified set; skipping self-calibration bootstrap")
    elif args.calibration is None:
        print("[run] no calibration artifact provided; attempting bootstrap self-calibration on first stereo pair")
    else:
        print("[run] self-calibration bootstrap disabled: using provided calibration artifact")

    receipts = ReceiptStore(Path(args.receipts_db))

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
    elif args.merge_profile == "calibrated_sgbm":
        merge_params = TemporalMergeParams(
            max_cost=56.0,
            min_gap=1.0,
            min_conf=0.12,
            tau_close_disp=2.0,
            conf_improvement_req=0.02,
            max_age_keep=16,
            min_stability_for_strong=1,
            smooth_alpha_strong=0.65,
            smooth_alpha_weak=0.45,
            min_evidence_frames=1,
            weak_conf_scale=0.6,
            region_min_pixels=40,
            region_max_disp_std=6.0,
            region_min_fill_ratio=0.12,
            region_score_enable=False,
            region_score_threshold_strong=0.60,
            region_score_threshold_weak=0.40,
            region_model_path=None,
        )
        default_diff_threshold = 12
    else:
        merge_params = TemporalMergeParams()
        default_diff_threshold = 12
    diff_threshold = args.diff_threshold if args.diff_threshold is not None else default_diff_threshold
    default_disp_max = 128 if args.merge_profile in {"calibrated", "calibrated_sgbm"} or args.calibration else 64
    disp_min = int(args.disp_min)
    disp_max = int(args.disp_max) if args.disp_max is not None else default_disp_max
    matcher = args.matcher
    if matcher == "auto":
        matcher = "opencv_sgbm" if args.merge_profile in {"calibrated", "calibrated_sgbm"} or args.calibration else "cpu_census"
    effective_merge_profile = args.merge_profile
    if matcher == "opencv_sgbm" and args.merge_profile == "calibrated":
        effective_merge_profile = "calibrated_sgbm"
        merge_params = TemporalMergeParams(
            max_cost=56.0,
            min_gap=1.0,
            min_conf=0.12,
            tau_close_disp=2.0,
            conf_improvement_req=0.02,
            max_age_keep=16,
            min_stability_for_strong=1,
            smooth_alpha_strong=0.65,
            smooth_alpha_weak=0.45,
            min_evidence_frames=1,
            weak_conf_scale=0.6,
            region_min_pixels=40,
            region_max_disp_std=6.0,
            region_min_fill_ratio=0.12,
            region_score_enable=False,
            region_score_threshold_strong=0.60,
            region_score_threshold_weak=0.40,
            region_model_path=None,
        )
    confidence_model_path = None
    confidence_model = None
    confidence_threshold = None
    if matcher == "opencv_sgbm":
        confidence_model_path = Path(args.confidence_model) if args.confidence_model else None
        if confidence_model_path is not None and confidence_model_path.exists():
            confidence_model = _load_confidence_model(confidence_model_path)
            confidence_threshold = float(confidence_model.get("best_threshold", {}).get("tau", merge_params.min_conf))
            merge_params.min_conf = confidence_threshold
    if args.region_min_pixels is not None:
        merge_params.region_min_pixels = int(args.region_min_pixels)
    if args.region_max_disp_std is not None:
        merge_params.region_max_disp_std = float(args.region_max_disp_std)
    if args.region_min_fill_ratio is not None:
        merge_params.region_min_fill_ratio = float(args.region_min_fill_ratio)
    if args.region_model:
        merge_params.region_score_enable = True
        merge_params.region_model_path = str(Path(args.region_model))
    if args.region_score_threshold_strong is not None:
        merge_params.region_score_threshold_strong = float(args.region_score_threshold_strong)
    if args.region_score_threshold_weak is not None:
        merge_params.region_score_threshold_weak = float(args.region_score_threshold_weak)
    roi_mode = "full_frame" if args.full_frame_roi else "motion"
    print(f"[run] roi mode: {roi_mode}")
    run_config = {
        "width": args.width,
        "height": args.height,
        "every_n": args.every_n,
        "sbs": args.sbs,
        "auto_res": args.auto_res,
        "start_seconds": args.start_seconds,
        "tile_size": args.tile_size,
        "tile_halo": args.tile_halo,
        "region_min_pixels": merge_params.region_min_pixels,
        "region_max_disp_std": merge_params.region_max_disp_std,
        "region_min_fill_ratio": merge_params.region_min_fill_ratio,
        "region_score_enable": merge_params.region_score_enable,
        "region_score_threshold_strong": merge_params.region_score_threshold_strong,
        "region_score_threshold_weak": merge_params.region_score_threshold_weak,
        "region_model_path": merge_params.region_model_path,
        "merge_profile": effective_merge_profile,
        "motion_roi": args.motion_roi,
        "full_frame_roi": args.full_frame_roi,
        "roi_mode": roi_mode,
        "diff_threshold": diff_threshold,
        "disp_min": disp_min,
        "disp_max": disp_max,
        "matcher": matcher,
        "sgbm_block_size": args.sgbm_block_size,
        "sgbm_uniqueness": args.sgbm_uniqueness,
        "expand_texture_min": int(args.expand_texture_min),
        "expand_lr_max": int(args.expand_lr_max),
        "expand_median_max": int(args.expand_median_max),
        "confidence_model": str(confidence_model_path) if confidence_model_path is not None else None,
        "debug_loose": args.debug_loose,
        "sweep": args.sweep,
        "max_frames": args.max_frames,
        "selfcal_mode": "pending",
    }
    run_id = receipts.create_run(
        source=src_path,
        calibration_id=None,
        config=run_config,
    )
    # Streamers are initialized before bootstrap calibration so we can use the first pair.
    calibration = None
    calibration_source = "none"
    if dual_file_mode:
        streamer_left = FrameStreamer(
            args.left_file,
            args.width,
            args.height,
            args.every_n,
            gray=True,
            youtube=False,
            start_seconds=args.start_seconds,
            auto_res=args.auto_res,
            status_prefix="[run]",
        )
        streamer_right = FrameStreamer(
            args.right_file,
            args.width,
            args.height,
            args.every_n,
            gray=True,
            youtube=False,
            start_seconds=args.start_seconds,
            auto_res=args.auto_res,
            status_prefix="[run]",
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
            auto_res=args.auto_res,
            status_prefix="[run]",
        )

    if args.calibration:
        try:
            calibration = load_calibration_artifact(Path(args.calibration))
            calibration_source = "provided"
            print(f"[run] loaded calibration artifact {calibration.calibration_id}")
            if args.force_unrectified:
                calibration = None
                calibration_source = "force_unrectified"
                print("[run] --force-unrectified overrides provided calibration")
        except Exception as exc:
            print(f"[run] failed to load provided calibration ({args.calibration}): {exc}")
            print("[run] will try bootstrap if possible")
    elif not args.force_unrectified:
        bootstrap_seed, bootstrap_result = _seed_and_self_calibrate(
            dual_file_mode=dual_file_mode,
            left_streamer=streamer_left if dual_file_mode else None,
            right_streamer=streamer_right if dual_file_mode else None,
            streamer=streamer,
            args=args,
            timeout=args.timeout,
            selfcal_output=selfcal_output,
            outdir=outdir,
            rig_id=args.selfcal_rig_id,
            max_features=args.selfcal_max_features,
            save_debug=args.selfcal_debug,
        )
        if bootstrap_result is None:
            bootstrap_error = RuntimeError("bootstrap self-calibration could not acquire a valid stereo pair")
            print("[run] self-calibration bootstrap failed: could not acquire a valid stereo pair")
            print("[run] continuing in unrectified demo mode")
        elif isinstance(bootstrap_result, Exception):
            bootstrap_error = bootstrap_result
            print(f"[run] self-calibration bootstrap failed: {bootstrap_result}")
            print("[run] continuing in unrectified demo mode")
        else:
            calibration = bootstrap_result
            calibration_source = "selfcal_bootstrap"
            print(f"[run] bootstrap self-calibration ready: {calibration.calibration_id}")
    if calibration is None:
        print("[run] no usable calibration artifact; running unrectified demo mode")

    if calibration is not None:
        receipts.conn.execute(
            "UPDATE runs SET calibration_id=?, config_json=? WHERE id=?",
            (
                calibration.calibration_id,
                json.dumps(
                    {
                        **run_config,
                        "selfcal_mode": calibration_source,
                    },
                    sort_keys=True,
                ),
                run_id,
            ),
        )
        receipts.conn.commit()
    else:
        receipts.conn.execute(
            "UPDATE runs SET config_json=? WHERE id=?",
            (json.dumps({**run_config, "selfcal_mode": calibration_source}, sort_keys=True), run_id),
        )
        receipts.conn.commit()

    print(f"[run] merge profile: {effective_merge_profile}")
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
                if bootstrap_seed is not None:
                    _, left, right = bootstrap_seed
                    bootstrap_seed = None
                else:
                    left, right = streamer_left.get_latest(), streamer_right.get_latest()
                    if left is None or right is None:
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
                if bootstrap_seed is not None:
                    source_frame, left, right = bootstrap_seed
                    bootstrap_seed = None
                else:
                    seed_frame = _wait_for_frame(streamer, args, last_seen)
                    if seed_frame is None:
                        print(f"[run] no more frames available after frame {frame_index}; stopping")
                        break
                    source_frame, left, right = seed_frame
                if left is None:
                    print(f"[run] failed to grab frame {frame_index} within {args.timeout}s")
                    sys.exit(1)
                last_seen = _packet_token(source_frame)

                if args.sbs:
                    mid = left.shape[1] // 2
                    sbs_frame = left
                    left = sbs_frame[:, :mid]
                    right = sbs_frame[:, mid:]

            stage = {}
            if not dual_file_mode and source_frame is not None:
                stage["source_selected_index"] = source_frame.get("selected_index")
                stage["source_pts_time"] = source_frame.get("pts_time")
            t_start = time.perf_counter()
            if calibration is not None:
                t_rect = time.perf_counter()
                left, right = rectify_pair(left, right, calibration)
                stage["rectify_ms"] = (time.perf_counter() - t_rect) * 1000.0

            t_roi = time.perf_counter()
            if args.full_frame_roi:
                roi = full_frame_roi(left, args.tile_size)
            else:
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
                if matcher != "cpu_census":
                    print("[run] threshold sweep currently supports only matcher=cpu_census")
                    sys.exit(1)
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
                                d_min=disp_min,
                                d_max=disp_max,
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
                block_size = None
                uniqueness_ratio = None
                disp = sweep_best["cand"]
                cost_min = sweep_best["cost"]
                conf = sweep_best["conf"]
                valid = sweep_best["valid"]
            else:
                if matcher == "opencv_sgbm":
                    min_tex = None
                    max_cost = None
                    min_conf = None
                    block_size = args.sgbm_block_size
                    uniqueness_ratio = args.sgbm_uniqueness
                    disp, cost_min, conf, valid, sgbm_evidence = stereo_sgbm_roi(
                        left,
                        right,
                        d_min=disp_min,
                        d_max=disp_max,
                        block_size=block_size,
                        uniqueness_ratio=uniqueness_ratio,
                        roi_mask=roi["roi_mask"],
                        expand_texture_min=args.expand_texture_min,
                        expand_lr_max=args.expand_lr_max,
                        expand_median_max=args.expand_median_max,
                    )
                else:
                    sgbm_evidence = None
                    block_size = None
                    uniqueness_ratio = None
                    if calibration is not None:
                        min_tex = 4
                        max_cost = 48
                        min_conf = 1
                    else:
                        min_tex = 4 if args.debug_loose else 8
                        max_cost = 48 if args.debug_loose else 30
                        min_conf = 1 if args.debug_loose else 2
                    disp, cost_min, conf, valid = stereo_census_roi(
                        left,
                        right,
                        d_min=disp_min,
                        d_max=disp_max,
                        min_tex=min_tex,
                        max_cost=max_cost,
                        min_conf=min_conf,
                        roi_mask=roi["roi_mask"],
                    )
            if frame_index == 0:
                if matcher == "opencv_sgbm":
                    print(
                        "[run] stereo matcher:"
                        f" matcher={matcher}"
                        f" disp_min={disp_min}"
                        f" disp_max={disp_max}"
                        f" block_size={block_size}"
                        f" uniqueness={uniqueness_ratio}"
                        f" conf_model={'yes' if confidence_model is not None else 'no'}"
                        f" conf_tau={merge_params.min_conf:.2f}"
                        f" calibrated={'yes' if calibration is not None else 'no'}"
                    )
                else:
                    print(
                        "[run] stereo matcher:"
                        f" matcher={matcher}"
                        f" disp_min={disp_min}"
                        f" disp_max={disp_max}"
                        f" min_tex={min_tex}"
                        f" max_cost={max_cost}"
                        f" min_conf={min_conf}"
                        f" calibrated={'yes' if calibration is not None else 'no'}"
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
            cand_conf_override = None
            if matcher == "opencv_sgbm" and confidence_model is not None and sgbm_evidence is not None:
                cand_conf_override = _apply_confidence_model(
                    confidence_model,
                    valid=valid.astype(np.uint8),
                    roi_mask=roi_mask,
                    disp_q8=(disp >> 8).astype(np.uint8),
                    cost_u8=np.clip(cand_cost_f, 0, 255).astype(np.uint8),
                    heuristic_conf_u8=np.clip(conf.astype(np.float32) * (255.0 / 31.0), 0, 255).astype(np.uint8),
                    lr_delta_u8=np.clip(sgbm_evidence["lr_delta"], 0, 255).astype(np.uint8),
                    median_delta_u8=np.clip(sgbm_evidence["median_delta"], 0, 255).astype(np.uint8),
                    texture_u8=np.clip(sgbm_evidence["texture"], 0, 255).astype(np.uint8),
                    disp_gradient_u8=np.clip(sgbm_evidence["disp_gradient"], 0, 255).astype(np.uint8),
                    edge_distance_u8=np.clip(sgbm_evidence["edge_distance"], 0, 255).astype(np.uint8),
                    border_penalty_u8=np.clip(sgbm_evidence["border_penalty"], 0, 255).astype(np.uint8),
                )

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
                cand_conf_override=cand_conf_override,
                evidence_maps=sgbm_evidence,
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
                valid.astype(np.uint8),
                np.clip(cand_cost_f, 0, 255).astype(np.uint8),
                np.clip(cand_conf * 255.0, 0, 255).astype(np.uint8),
                np.clip((sgbm_evidence["lr_delta"] if sgbm_evidence is not None else np.full_like(valid, 255)), 0, 255).astype(np.uint8),
                np.clip((sgbm_evidence["median_delta"] if sgbm_evidence is not None else np.full_like(valid, 255)), 0, 255).astype(np.uint8),
                np.clip((sgbm_evidence["texture"] if sgbm_evidence is not None else np.zeros_like(valid)), 0, 255).astype(np.uint8),
                np.clip((sgbm_evidence["disp_gradient"] if sgbm_evidence is not None else np.full_like(valid, 255)), 0, 255).astype(np.uint8),
                np.clip((sgbm_evidence["edge_distance"] if sgbm_evidence is not None else np.zeros_like(valid)), 0, 255).astype(np.uint8),
                np.clip((sgbm_evidence["border_penalty"] if sgbm_evidence is not None else np.full_like(valid, 255)), 0, 255).astype(np.uint8),
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
                    "tau_cost": merge_params.max_cost,
                    "tau_conf": merge_params.min_conf,
                    "min_tex": min_tex,
                    "min_gap": merge_params.min_gap,
                    "tau_close_disp": merge_params.tau_close_disp,
                    "min_evidence_frames": merge_params.min_evidence_frames,
                    "weak_conf_scale": merge_params.weak_conf_scale,
                    "region_min_pixels": merge_params.region_min_pixels,
                    "region_max_disp_std": merge_params.region_max_disp_std,
                    "region_min_fill_ratio": merge_params.region_min_fill_ratio,
                    "region_score_enable": merge_params.region_score_enable,
                    "region_score_threshold_strong": merge_params.region_score_threshold_strong,
                    "region_score_threshold_weak": merge_params.region_score_threshold_weak,
                    "region_model_path": merge_params.region_model_path,
                    "merge_profile": effective_merge_profile,
                    "matcher": matcher,
                    "expand_texture_min": int(args.expand_texture_min),
                    "expand_lr_max": int(args.expand_lr_max),
                    "expand_median_max": int(args.expand_median_max),
                    "confidence_model": str(confidence_model_path) if confidence_model_path is not None else None,
                    "confidence_threshold": confidence_threshold,
                },
                counts={
                    "candidate_valid_px": int(valid.sum()),
                    "promoted": merge_stats["accepted_pixels"],
                    "promoted_close": merge_stats["accepted_close_pixels"],
                    "promoted_better": merge_stats["accepted_better_pixels"],
                    "promoted_temporal": merge_stats["accepted_temporal_pixels"],
                    "expired": merge_stats["expired_pixels"],
                    "region_components_total": merge_stats.get("region_components_total", 0),
                    "region_components_kept": merge_stats.get("region_components_kept", 0),
                    "region_components_rejected": merge_stats.get("region_components_rejected", 0),
                    "region_rejected_pixels": merge_stats.get("region_rejected_pixels", 0),
                    "region_strong_kept": merge_stats.get("region_strong_kept", 0),
                    "region_weak_kept": merge_stats.get("region_weak_kept", 0),
                    "roi_pixels": int(roi["roi_mask"].sum()),
                },
                residual_mean=float(accepted_costs.mean()) if accepted_costs.size else 0.0,
                residual_p95=float(np.percentile(accepted_costs, 95)) if accepted_costs.size else 0.0,
                invariants={
                    "valid_mask_nonzero": bool(valid.any()),
                    "roi_nonempty": bool(roi["roi_mask"].any()),
                    "calibrated": calibration is not None,
                    "calibration_source": calibration_source,
                    "calibration_id": calibration.calibration_id if calibration else None,
                    "calibration_schema": calibration.metadata.get("schema_version") if calibration else None,
                },
            )
            for kind, name in [
                ("left_input", f"left_input_f{frame_index:04d}.png"),
                ("right_input", f"right_input_f{frame_index:04d}.png"),
                ("roi_mask", f"roi_mask_f{frame_index:04d}.png"),
                ("candidate_disp", f"candidate_disp_f{frame_index:04d}.png"),
                ("candidate_mask", f"candidate_mask_f{frame_index:04d}.png"),
                ("candidate_cost", f"candidate_cost_f{frame_index:04d}.png"),
                ("candidate_conf", f"candidate_conf_f{frame_index:04d}.png"),
                ("candidate_lr_delta", f"candidate_lr_delta_f{frame_index:04d}.png"),
                ("candidate_median_delta", f"candidate_median_delta_f{frame_index:04d}.png"),
                ("candidate_texture", f"candidate_texture_f{frame_index:04d}.png"),
                ("candidate_disp_gradient", f"candidate_disp_gradient_f{frame_index:04d}.png"),
                ("candidate_edge_distance", f"candidate_edge_distance_f{frame_index:04d}.png"),
                ("candidate_border_penalty", f"candidate_border_penalty_f{frame_index:04d}.png"),
                ("canonical_disp", f"canonical_disp_f{frame_index:04d}.png"),
                ("promoted_depth_npz", f"promoted_depth_f{frame_index:04d}.npz"),
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
    except KeyboardInterrupt:
        print("[run] interrupted; shutting down")
    finally:
        receipts.close()
        if dual_file_mode:
            streamer_left.close()
            streamer_right.close()
        else:
            streamer.close()


if __name__ == "__main__":
    main()
