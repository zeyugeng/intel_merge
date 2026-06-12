# Intel Cup · 鸟类活动 AI 监测员（合并版）

声源定位（ODAS）+ 鸟类声学识别（BirdNET）+ 视觉检测（YOLO）的声视融合监测系统。

## 目录

| 目录 | 说明 |
|------|------|
| `PythonProject/` | Python 主程序：融合、BirdNET、YOLO、GUI |
| `odas/` | ODAS 声源定位（含 Yundea 14 麦阵列配置与测试脚本） |

## 快速开始

### Python 项目

```bash
cd PythonProject
conda env create -f environment.yml
conda activate intel_cup
python scripts/run_fusion.py
```

详见 [PythonProject/README.md](PythonProject/README.md)。

### ODAS 单独测试

```bash
cd odas
./scripts/run_odas_test.sh
```

## 上游来源

- Python 部分基于 [zeyugeng/intel-cup](https://github.com/zeyugeng/intel-cup)
- ODAS 基于 [introlab/odas](https://github.com/introlab/odas)

## 许可证

各子目录保留原项目许可证（MIT）。
