#!/usr/bin/env bash
# 修复 Yundea 麦克风常见权限/驱动问题 (需 sudo)
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "请使用 sudo 运行: sudo $0"
    exit 1
fi

TARGET_USER="${SUDO_USER:-$USER}"
RULE="/etc/udev/rules.d/99-yundea-micarray.rules"

echo "=== 修复 Yundea 16P14MICARRAY ==="

# 1. audio 组
if id -nG "$TARGET_USER" | grep -qw audio; then
    echo "[OK] $TARGET_USER 已在 audio 组"
else
    usermod -aG audio "$TARGET_USER"
    echo "[完成] 已将 $TARGET_USER 加入 audio 组 (需重新登录生效)"
fi

# 2. 禁用 Mass Storage 接口 (释放复合设备资源)
cat > "$RULE" << 'EOF'
# Yundea 16P14MICARRAY: 禁用 Mass Storage 接口，仅保留音频
SUBSYSTEM=="usb", ATTR{idVendor}=="4654", ATTR{idProduct}=="3e41", ENV{INTERFACE}=="8/6/80", ATTR{authorized}="0"
EOF
echo "[完成] 写入 udev 规则: $RULE"

udevadm control --reload-rules
udevadm trigger

# 3. 尝试立即 unbind 已绑定的 storage
for dev in /sys/bus/usb/devices/*; do
    [[ -f "$dev/idVendor" && "$(cat "$dev/idVendor")" == "4654" ]] || continue
    [[ -f "$dev/idProduct" && "$(cat "$dev/idProduct")" == "3e41" ]] || continue
    for iface in "$dev"/*:*; do
        [[ -f "$iface/bInterfaceClass" && "$(cat "$iface/bInterfaceClass")" == "08" ]] || continue
        if [[ -L "$iface/driver" ]]; then
            name=$(basename "$iface")
            echo "$name" > /sys/bus/usb/drivers/usb-storage/unbind 2>/dev/null && \
                echo "[完成] 已 unbind usb-storage: $name" || \
                echo "[跳过] 无法 unbind $name (可能已禁用)"
        fi
    done
done

echo
echo "请重新插拔麦克风 USB 线，然后运行:"
echo "  ./scripts/diagnose_mic.sh"
