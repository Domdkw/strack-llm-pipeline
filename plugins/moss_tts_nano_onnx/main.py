from onnx_tts_runtime import OnnxTtsRuntime
from utils import MODEL_DIR
from configs import read_setting
from plugin_mount import ModelInstance
import os

class MossTtsNanoInstance(ModelInstance, OnnxTtsRuntime):
    def __init__(self, model_info: dict, tts_model_dir: str, codec_model_dir: str, thread: int):
        super().__init__(model_info)
        OnnxTtsRuntime.__init__(
            self,
            tts_model_dir=tts_model_dir,
            codec_model_dir=codec_model_dir,
            thread_count=thread,
        )
    @property
    def model_name(self) -> str:
        return 'MOSS-TTS-Nano-100M'
    @property
    def model_type(self) -> str:
        return 'tts'
    @property
    def runtime(self) -> str:
        return 'onnxruntime'

def create_instance(_model_info: dict):
    tts_model_dir: str = os.path.join(MODEL_DIR, 'MOSS-TTS-Nano-100M-ONNX')
    codec_model_dir: str = os.path.join(MODEL_DIR, 'MOSS-Audio-Tokenizer-Nano-ONNX')
    thread: int = read_setting("llama_params")["n_threads"]
    moss_instance = MossTtsNanoInstance(
        model_info=_model_info,
        tts_model_dir=tts_model_dir,
        codec_model_dir=codec_model_dir,
        thread=thread,
    )
    return moss_instance
