import os
# 必须放在 import birdnet 之前,屏蔽冗余日志
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from pathlib import Path
import birdnet

class SoundPredict:
    def __init__(self, species_list=None):
        self.species_list = species_list
        self.model = None
        self.session = None
        self._ctx = None

    def load_model(self):
        if self.session:
            return
        print("正在加载 BirdNET 模型...")
        self.model = birdnet.load("acoustic", "2.4", "tf")
        self._ctx = self.model.predict_session(
            custom_species_list=self.species_list
        )
        self.session = self._ctx.__enter__()
        print("模型加载完成")

    def predict(self, audio_file):
        if not self.session:
            raise RuntimeError("请先调用 load_model() 加载模型")
        audio_file = str(Path(audio_file))
        return self.session.run([audio_file])

    def close(self):
        if self._ctx:
            self._ctx.__exit__(None, None, None)

        self.model = None
        self.session = None
        self._ctx = None

    #兼容with写法
    def __enter__(self):
        self.load_model()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == "__main__":
    #multiprocessing.freeze_support()
    predictor = SoundPredict(species_list="species_list.txt")

    predictor.load_model()

    result1 = predictor.predict("soundscape.wav")
    print(result1.to_structured_array())

    result2 = predictor.predict("soundscape.wav")
    print(result2.to_structured_array())

    predictor.close()

