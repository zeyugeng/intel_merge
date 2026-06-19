# Intel Cup · 鸟类活动 AI 监测员（合并版）

声源定位（ODAS）+ 鸟类声学识别（BirdNET）+ 视觉检测（YOLO）的声视融合监测系统。

## 目录

| 目录 | 说明 |
|------|------|
| `PythonProject/` | Python 主程序：融合、BirdNET、YOLO、GUI |
| `odas/` | ODAS 声源定位（含 Yundea 14 麦阵列配置与测试脚本） |

## 快速开始

### 单终端声源云台跟踪（推荐）

```bash
cd PythonProject
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-sound-ptz.txt
python scripts/run_sound_ptz_all.py --no-preview
```

详见 [PythonProject/README.md](PythonProject/README.md)。

`run_sound_ptz_all.py` 会在一个终端内启动 ODAS、TCP 桥接和云台跟踪。若需要 RTSP 预览，去掉 `--no-preview`。

### ODAS 单独测试

```bash
cd odas
./scripts/run_odas_test.sh
```

### 可选功能

```bash
cd PythonProject
source .venv/bin/activate

# 视觉检测 / 声视融合
pip install -r requirements-vision.txt
python scripts/run_visual.py
python scripts/run_fusion.py

# BirdNET 离线鸟声识别
pip install -r requirements-birdnet.txt
python scripts/run_birdnet.py data/audio/your.wav
```

## 上游来源

- Python 部分基于 [zeyugeng/intel-cup](https://github.com/zeyugeng/intel-cup)
- ODAS 基于 [introlab/odas](https://github.com/introlab/odas)

## 许可证

各子目录保留原项目许可证（MIT）。
