from pathlib import Path
import logging
from onnx_tts_runtime import OnnxTtsRuntime
from utils import MODEL_DIR
from configs import read_setting
from plugin_mount import ModelInstance
import os

logger = logging.getLogger(__name__)

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
    def synthesize(
        self,
        *,
        text: str,
        prompt_audio_path: str | Path,
        output_audio_path: str | Path | None = None,
        max_new_frames: int | None = None,
        enable_normalize_tts_text: bool = True,
        voice_clone_max_text_tokens: int = 75,
        seed: int | None = None,
    ) -> str:
        """ 合成语音 """
        return super().synthesize(
            text=text,
            prompt_audio_path=prompt_audio_path,
            output_audio_path=output_audio_path,
            max_new_frames=max_new_frames,
            enable_normalize_tts_text=enable_normalize_tts_text,
            voice_clone_max_text_tokens=voice_clone_max_text_tokens,
            seed=seed,
        )

def create_instance(_model_info: dict):
    tts_model_dir: str = os.path.join(MODEL_DIR, 'MOSS-TTS-Nano-100M-ONNX')
    codec_model_dir: str = os.path.join(MODEL_DIR, 'MOSS-Audio-Tokenizer-Nano-ONNX')
    if not os.path.isdir(tts_model_dir):
        raise FileNotFoundError(f"TTS model dir not found: {tts_model_dir}")
    if not os.path.isdir(codec_model_dir):
        raise FileNotFoundError(f"TTS codec model dir not found: {codec_model_dir}")
    thread: int = read_setting("llama_params")["n_threads"]
    logger.info("初始化 MOSS-TTS-Nano tts_dir=%s codec_dir=%s thread=%s", tts_model_dir, codec_model_dir, thread)
    moss_instance = MossTtsNanoInstance(
        model_info=_model_info,
        tts_model_dir=tts_model_dir,
        codec_model_dir=codec_model_dir,
        thread=thread,
    )
    return moss_instance
