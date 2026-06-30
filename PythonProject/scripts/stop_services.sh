#!/usr/bin/env bash
# 释放 run_sound_ptz_all / ODAS 占用的 5000、9001 端口
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== 停止鸟类监测相关进程 ==="

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON="${ROOT}/.venv/bin/python"
else
    PYTHON="python3"
fi

"${PYTHON}" "${ROOT}/scripts/free_ports.py"
exit $?
