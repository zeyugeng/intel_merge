#!/usr/bin/env bash
# 单独测试 ODAS 声源定位（终端 JSON 输出）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="${ROOT}/build/lib:${LD_LIBRARY_PATH:-}"
CFG="${1:-${ROOT}/config/myArray.cfg}"
ODAS="${ROOT}/build/bin/odaslive"

echo "=== ODAS 单独测试 ==="
echo "配置: ${CFG}"
echo "库路径: ${ROOT}/build/lib"
echo

if [[ ! -x "${ODAS}" ]]; then
    echo "错误: 未找到 ${ODAS}，请先在 odas/build 目录执行 cmake && make"
    exit 1
fi

if ! arecord -l 2>/dev/null | grep -q "Y16P14MICARRAY"; then
    echo "警告: 未检测到 Yundea 16P14MICARRAY，请确认 USB 麦克风已连接"
    arecord -l || true
    echo
fi

echo "麦克风预检 (14 通道 @ 32kHz, 2 秒)..."
if arecord -D hw:1,0 -f S16_LE -r 32000 -c 14 \
    --period-time=1000 --buffer-time=2000 -d 2 /tmp/odas_mic_test.wav 2>/tmp/odas_mic_test.err; then
    echo "麦克风录音 OK: /tmp/odas_mic_test.wav ($(stat -c%s /tmp/odas_mic_test.wav) bytes)"
else
    echo "麦克风录音失败 (ODAS 也无法启动，请先修复):"
    sed 's/^/  /' /tmp/odas_mic_test.err 2>/dev/null || true
    echo
    echo "  ENOSPC = USB 等时带宽不足 (不是磁盘满)。当前麦克风接在 Hub 上，且与摄像头/WiFi 共用 Bus 003。"
    echo "  请执行:"
    echo "    1. 拔掉 USB 摄像头"
    echo "    2. 麦克风直插主板 USB 3.0 口 (Bus 004 空闲)"
    echo "    3. ./scripts/diagnose_mic.sh   # 详细诊断"
    echo "    4. sudo ./scripts/fix_mic_array.sh   # 修复权限/禁用 Mass Storage"
    echo
    exit 1
fi

echo
echo "启动 ODAS (Ctrl+C 退出，对着麦克风发声观察 JSON 输出)..."
echo "---"
exec "${ODAS}" -c "${CFG}"
