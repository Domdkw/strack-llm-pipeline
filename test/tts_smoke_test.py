"""MOSS-TTS-Nano ONNX 推理冒烟测试。

用手动指定的 TTS / codec 两个模型目录 + test/prompt_audio.mp3 参考音频
执行音色克隆，输出到 test/tts_output.wav。

用法：
    python test/tts_smoke_test.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = REPO_ROOT / "plugins" / "moss_tts_nano_onnx"
TTS_DIR = REPO_ROOT / "models" / "MOSS-TTS-Nano-100M-ONNX"
CODEC_DIR = REPO_ROOT / "models" / "MOSS-Audio-Tokenizer-Nano-ONNX"
PROMPT_AUDIO = REPO_ROOT / "test" / "prompt_audio.mp3"
OUTPUT_AUDIO = REPO_ROOT / "test" / "tts_output.wav"

sys.path.insert(0, str(PLUGIN_DIR))
from onnx_tts_runtime import OnnxTtsRuntime


def main() -> None:
    runtime = OnnxTtsRuntime(
        tts_model_dir=TTS_DIR,
        codec_model_dir=CODEC_DIR,
        thread_count=4,
    )
    result: dict[str, Any] = runtime.synthesize(
        text="""有一个国家，屹立在世界的东方，那是伟大的中国;有一面国旗，闪耀着五星的光芒，那是中国的五星红旗。我的一切梦想，连着所有中国人的梦，汇成了伟大的中国梦，向着梦想，前进的步伐从未停止，向国旗敬礼!""",
        prompt_audio_path=PROMPT_AUDIO,
        output_audio_path=OUTPUT_AUDIO,
        seed=1,
    )
    print(
        "SMOKE_OK",
        result["audio_path"],
        result["sample_rate"],
        result["waveform"].shape,
        result["text_chunks"],
    )


if __name__ == "__main__":
    main()
