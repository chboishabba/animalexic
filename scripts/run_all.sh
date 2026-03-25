#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[all] Configuring & building shaders"
cmake -S "$ROOT" -B "$ROOT/build" >/dev/null
cmake --build "$ROOT/build" --target shaders

echo "[all] Running CPU harness"
python "$ROOT/scripts/test_stereo_roi_cpu.py"

echo "[all] Descriptor layout / Vulkan availability check"
python "$ROOT/scripts/run_stereo_stub.py" || true

echo "[all] Attempting Vulkan pipeline creation (will exit cleanly if ICD missing)"
python "$ROOT/scripts/vk_stereo_pipeline.py" || true

echo "[all] Done."
