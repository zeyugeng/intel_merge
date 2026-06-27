from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / "core"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
DATASETS_DIR = DATA_DIR / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "output"
CAPTURES_DIR = OUTPUT_DIR / "captures"
RUNS_DIR = OUTPUT_DIR / "runs"
ASSETS_DIR = PROJECT_ROOT / "assets"
DOCS_DIR = PROJECT_ROOT / "docs"

YOLO_MODEL_PATH = MODELS_DIR / "yolo26n.pt"
YOLO_OPENVINO_DIR = MODELS_DIR / "yolo26n_openvino_model"
DEFAULT_AUDIO_PATH = AUDIO_DIR / "soundscape.wav"
