#!/usr/bin/env bash
# 诊断 Yundea 16P14MICARRAY 录音失败原因
set -uo pipefail

echo "=== 麦克风阵列诊断 ==="
echo

echo "[1] 用户组 (需在 audio 组):"
groups
if groups | grep -qw audio; then
    echo "  OK: 已在 audio 组"
else
    echo "  缺失: 不在 audio 组，请执行:"
    echo "    sudo usermod -aG audio \$USER"
    echo "    然后注销并重新登录"
fi
echo

echo "[2] 声卡:"
arecord -l 2>&1 | grep -E "card|Y16P14|Yundea" || arecord -l 2>&1
echo

echo "[3] USB 拓扑 (麦克风不应接 Hub，且避免与摄像头/WiFi 同总线):"
lsusb -t 2>&1
echo

if lsusb -t 2>&1 | grep -q "Hub.*480M" && lsusb -t 2>&1 | grep -B5 "Mass Storage\|Audio" | grep -q "Hub"; then
    echo "  警告: 麦克风很可能接在外置 USB Hub 上"
fi
if lsusb -t 2>&1 | grep -q "Video.*480M"; then
    echo "  警告: 同总线上有 USB 摄像头，会占用大量等时带宽"
fi
if lsusb -t 2>&1 | grep -q "802.11\|WLAN\|rtl8"; then
    echo "  警告: 同总线上有 USB WiFi 网卡"
fi
echo "  建议: 将麦克风直插主板 USB 3.0 口 (Bus 004 目前空闲)"
echo

echo "[4] 设备带宽需求:"
lsusb -d 4654:3e41 -v 2>/dev/null | grep -E "idProduct|MaxPacketSize|bNrChannels|tSamFreq" | head -6 || echo "  (无法读取 USB 详情)"
echo "  14ch@32kHz 约需 924 bytes/ms USB 等时带宽 (USB 1.1 设备)"
echo

echo "[5] 录音测试:"
if arecord -D hw:1,0 -f S16_LE -r 32000 -c 14 \
    --period-time=1000 --buffer-time=2000 -d 1 /tmp/odas_diag.wav 2>/tmp/odas_diag.err; then
    echo "  OK: 录音成功 → /tmp/odas_diag.wav"
    rm -f /tmp/odas_diag.wav
else
    echo "  失败:"
    sed 's/^/    /' /tmp/odas_diag.err 2>/dev/null || true
    if grep -qi "nospc\|空间" /tmp/odas_diag.err 2>/dev/null; then
        echo
        echo "  结论: ENOSPC = USB 带宽不足，不是磁盘满了。"
        echo "  请按顺序尝试:"
        echo "    1. 拔掉 USB 摄像头"
        echo "    2. 麦克风改插主板后置 USB 3.0 口 (不要经过 Hub)"
        echo "    3. sudo usermod -aG audio \$USER 并重新登录"
        echo "    4. 可选: sudo ./scripts/fix_mic_array.sh 解除 Mass Storage 接口占用"
    fi
fi
