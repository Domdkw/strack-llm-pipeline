from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import soundfile as sf

from ort_cpu_runtime import OrtCpuRuntime
from text_normalization_pipeline import prepare_tts_request_texts

PLUGIN_DIR = Path(__file__).resolve().parent
REPO_ROOT = PLUGIN_DIR.parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "models"
DEFAULT_INTER_CHUNK_PAUSE_SHORT_SECONDS = 0.40
DEFAULT_INTER_CHUNK_PAUSE_LONG_SECONDS = 0.24
SENTENCE_END_PUNCTUATION = set(".!?。！？；;")
CLAUSE_SPLIT_PUNCTUATION = set(",，、；;：:")
CLOSING_PUNCTUATION = set("\"'”’)]}）】》」』")
MANIFEST_CANDIDATE_RELATIVE_PATHS = (
    "browser_poc_manifest.json",
    "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
    "MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json",
)


def ensure_model_dir(model_dir: str | Path | None = None) -> Path:
    """校验并返回包含 browser_poc_manifest.json 的模型目录。"""
    resolved_model_dir = (
        DEFAULT_MODEL_DIR.expanduser().resolve()
        if model_dir is None
        else Path(model_dir).expanduser().resolve()
    )
    if any((resolved_model_dir / item).is_file() for item in MANIFEST_CANDIDATE_RELATIVE_PATHS):
        return resolved_model_dir
    tried_paths = ", ".join(str((resolved_model_dir / item).resolve()) for item in MANIFEST_CANDIDATE_RELATIVE_PATHS)
    raise FileNotFoundError(f"browser_onnx model assets not found. tried: {tried_paths}")


def _contains_cjk(text: str) -> bool:
    """判断文本是否包含中文、日文假名或韩文。"""
    for character in str(text or ""):
        if (
            "\u4e00" <= character <= "\u9fff"
            or "\u3400" <= character <= "\u4dbf"
            or "\u3040" <= character <= "\u30ff"
            or "\uac00" <= character <= "\ud7af"
        ):
            return True
    return False


def _prepare_text_for_sentence_chunking(text: str) -> str:
    """清理空白并按语言补齐句末标点，为后续分句做准备。"""
    normalized_text = str(text or "").strip()
    if not normalized_text:
        raise ValueError("Text prompt cannot be empty.")
    normalized_text = normalized_text.replace("\r", " ").replace("\n", " ")
    while "  " in normalized_text:
        normalized_text = normalized_text.replace("  ", " ")
    if _contains_cjk(normalized_text):
        if normalized_text[-1] not in SENTENCE_END_PUNCTUATION:
            normalized_text += "。"
        return normalized_text
    if normalized_text[:1].islower():
        normalized_text = normalized_text[:1].upper() + normalized_text[1:]
    if normalized_text[-1].isalnum():
        normalized_text += "."
    if len([item for item in normalized_text.split() if item]) < 5:
        normalized_text = f"        {normalized_text}"
    return normalized_text


def _split_text_by_punctuation(text: str, punctuation: set[str]) -> list[str]:
    """按给定标点切分文本，紧跟标点的右引号/括号会保留在句内。"""
    sentences: list[str] = []
    current_chars: list[str] = []
    index = 0
    normalized_text = str(text or "")
    while index < len(normalized_text):
        character = normalized_text[index]
        current_chars.append(character)
        if character in punctuation:
            lookahead = index + 1
            while lookahead < len(normalized_text) and normalized_text[lookahead] in CLOSING_PUNCTUATION:
                current_chars.append(normalized_text[lookahead])
                lookahead += 1
            sentence = "".join(current_chars).strip()
            if sentence:
                sentences.append(sentence)
            current_chars.clear()
            while lookahead < len(normalized_text) and normalized_text[lookahead].isspace():
                lookahead += 1
            index = lookahead
            continue
        index += 1
    tail = "".join(current_chars).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _join_sentence_parts(left: str, right: str) -> str:
    """拼接两个句子片段；中文直接相连，西文之间补一个空格。"""
    if not left:
        return right
    if not right:
        return left
    if _contains_cjk(left) or _contains_cjk(right):
        return left + right
    return f"{left} {right}"


def _merge_audio_channels(channel_arrays: list[np.ndarray]) -> np.ndarray:
    """把按声道分开的采样数组堆叠为 (samples, channels) 的波形。"""
    if not channel_arrays:
        return np.zeros((0, 1), dtype=np.float32)
    if len(channel_arrays) == 1:
        return np.asarray(channel_arrays[0], dtype=np.float32).reshape(-1, 1)
    min_length = min(int(channel.shape[0]) for channel in channel_arrays)
    trimmed = [np.asarray(channel[:min_length], dtype=np.float32) for channel in channel_arrays]
    return np.stack(trimmed, axis=1)


def _concat_waveforms(waveforms: list[np.ndarray]) -> np.ndarray:
    """按时间轴拼接多段波形，自动跳过空片段。"""
    if not waveforms:
        return np.zeros((0, 1), dtype=np.float32)
    non_empty = [waveform for waveform in waveforms if waveform.size > 0]
    if not non_empty:
        channel_count = int(waveforms[0].shape[1]) if waveforms[0].ndim == 2 and waveforms[0].shape[1] > 0 else 1
        return np.zeros((0, channel_count), dtype=np.float32)
    return np.concatenate(non_empty, axis=0)


def _write_waveform_to_wav(path: str | Path, waveform: np.ndarray, sample_rate: int) -> Path:
    """将 [-1, 1] 浮点波形量化为 16bit PCM 并写入 WAV 文件。"""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = np.round(clipped * 32767.0).astype(np.int16)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(int(pcm16.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())
    return output_path


def _resample_linear(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """override 把 (frames, channels) 波形线性插值到目标采样率。"""
    if source_rate == target_rate or waveform.shape[0] == 0:
        return waveform
    target_length = max(1, int(round(waveform.shape[0] * target_rate / float(source_rate))))
    source_positions = np.arange(waveform.shape[0], dtype=np.float64)
    target_positions = np.linspace(0, waveform.shape[0] - 1, target_length, dtype=np.float64)
    channels = [
        np.interp(target_positions, source_positions, waveform[:, channel_index])
        for channel_index in range(waveform.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32, copy=False)


class OnnxTtsRuntime(OrtCpuRuntime):
    """ONNX CPU 推理封装：负责分词、参考音频编码、长文本分块与 WAV 输出。"""

    def __init__(
        self,
        *,
        tts_model_dir: str | Path | None = None,
        codec_model_dir: str | Path | None = None,
        thread_count: int,
        max_new_frames: int | None = None,
    ) -> None:
        """加载模型目录、创建输出目录并初始化 SentencePiece 分词器。

        可通过 tts_model_dir / codec_model_dir 手动指定 TTS 与 codec 两个模型的目录；
        未指定时沿用 model_dir 下的默认布局。
        model_dir 省略时必须提供 tts_model_dir（其目录内需含 browser_poc_manifest.json）。
        """
        super().__init__(
            tts_model_dir=tts_model_dir,
            codec_model_dir=codec_model_dir,
            thread_count=thread_count,
            max_new_frames=max_new_frames,
        )
        self.sp_model = spm.SentencePieceProcessor(model_file=str(self.tokenizer_path))

    def encode_text(self, text: str) -> list[int]:
        """把文本编码为 SentencePiece token id 列表。"""
        return [int(token_id) for token_id in self.sp_model.encode(str(text or ""), out_type=int)]

    def count_text_tokens(self, text: str) -> int:
        """统计文本的 token 数量。"""
        return len(self.encode_text(text))

    def prepare_synthesis_text(self, *, text: str, enable_normalize_tts_text: bool = True) -> dict[str, Any]:
        """对输入文本做 robust 清洗，返回清洗后文本与所用方法。"""
        return prepare_tts_request_texts(
            text=text,
            enable_normalize_tts_text=bool(enable_normalize_tts_text),
        )

    def split_text_by_token_budget(self, text: str, max_tokens: int) -> list[str]:
        """用二分查找按 token 上限切分文本，并尽量落在标点或空格处。"""
        remaining_text = str(text or "").strip()
        if not remaining_text:
            return []
        pieces: list[str] = []
        preferred_boundary_chars = CLAUSE_SPLIT_PUNCTUATION | SENTENCE_END_PUNCTUATION | {" "}
        while remaining_text:
            if self.count_text_tokens(remaining_text) <= max_tokens:
                pieces.append(remaining_text)
                break
            low = 1
            high = len(remaining_text)
            best_prefix_length = 1
            while low <= high:
                middle = (low + high) // 2
                candidate = remaining_text[:middle].strip()
                if not candidate:
                    low = middle + 1
                    continue
                if self.count_text_tokens(candidate) <= max_tokens:
                    best_prefix_length = middle
                    low = middle + 1
                else:
                    high = middle - 1
            cut_index = best_prefix_length
            prefix = remaining_text[:best_prefix_length]
            preferred_index = -1
            scan_min = max(-1, len(prefix) - 25)
            for scan_index in range(len(prefix) - 1, scan_min, -1):
                if prefix[scan_index] in preferred_boundary_chars:
                    preferred_index = scan_index + 1
                    break
            if preferred_index > 0:
                cut_index = preferred_index
            piece = remaining_text[:cut_index].strip()
            if not piece:
                piece = remaining_text[:best_prefix_length].strip()
                cut_index = best_prefix_length
            pieces.append(piece)
            remaining_text = remaining_text[cut_index:].strip()
        return pieces

    def split_voice_clone_text(self, text: str, max_tokens: int = 75) -> list[str]:
        """按句、子句、token 预算逐级切分，返回文本块。"""
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return []
        safe_max_tokens: int = max(1, int(max_tokens))
        prepared_text: str = _prepare_text_for_sentence_chunking(normalized_text)
        sentence_candidates: list[str] = _split_text_by_punctuation(prepared_text, SENTENCE_END_PUNCTUATION) or [prepared_text.strip()]
        sentence_slices: list[tuple[int, str]] = []
        for sentence_text in sentence_candidates:
            normalized_sentence: str = sentence_text.strip()
            if not normalized_sentence:
                continue
            sentence_token_count: int = self.count_text_tokens(normalized_sentence)
            # 添主句
            if sentence_token_count <= safe_max_tokens:
                sentence_slices.append((sentence_token_count, normalized_sentence))
                continue
            # 过大，按子句切分
            clause_candidates: list[str] = _split_text_by_punctuation(normalized_sentence, CLAUSE_SPLIT_PUNCTUATION)
            if len(clause_candidates) <= 1:
                clause_candidates = [normalized_sentence]
            for clause_text in clause_candidates:
                normalized_clause: str = clause_text.strip()
                if not normalized_clause:
                    continue
                clause_token_count: int = self.count_text_tokens(normalized_clause)
                if clause_token_count <= safe_max_tokens:
                    sentence_slices.append((clause_token_count, normalized_clause))
                    continue
                # 过大，按 token切分
                for piece in self.split_text_by_token_budget(normalized_clause, safe_max_tokens):
                    normalized_piece: str = piece.strip()
                    if normalized_piece:
                        sentence_slices.append((self.count_text_tokens(normalized_piece), normalized_piece))
        chunks: list[str] = []
        current_chunk: str = ""
        current_chunk_token_count: int = 0
        for sentence_token_count, sentence_text in sentence_slices:
            if not current_chunk:
                current_chunk: str = sentence_text
                current_chunk_token_count: int = sentence_token_count
                continue
            if current_chunk_token_count + sentence_token_count > safe_max_tokens:
                chunks.append(current_chunk.strip())
                current_chunk: str = sentence_text
                current_chunk_token_count: int = sentence_token_count
            else:
                current_chunk: str = _join_sentence_parts(current_chunk, sentence_text)
                current_chunk_token_count: int = self.count_text_tokens(current_chunk)
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks if len(chunks) > 1 else [normalized_text]
    
    def estimate_inter_chunk_pause_seconds(self, text_chunk: str) -> float:
        """根据文本词数返回块间停顿秒数，短句用更长的停顿。"""
        word_count = len([item for item in str(text_chunk or "").strip().split() if item])
        return (
            DEFAULT_INTER_CHUNK_PAUSE_SHORT_SECONDS
            if word_count <= 4
            else DEFAULT_INTER_CHUNK_PAUSE_LONG_SECONDS
        )

    def _load_reference_audio(self, reference_audio_path: str | Path) -> np.ndarray:
        """读取参考音频，重采样并调整声道，返回 (1, channels, samples) 波形。"""
        waveform, source_rate = sf.read(
            str(Path(reference_audio_path).expanduser().resolve()),
            dtype="float32",
            always_2d=True,
        )
        target_sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
        target_channels = int(self.codec_meta["codec_config"]["channels"])
        waveform = _resample_linear(np.asarray(waveform, dtype=np.float32), int(source_rate), target_sample_rate)
        current_channels = int(waveform.shape[1])
        if current_channels == target_channels:
            pass
        elif current_channels == 1 and target_channels > 1:
            waveform: np.ndarray = np.repeat(waveform, target_channels, axis=1)
        elif current_channels > 1 and target_channels == 1:
            waveform: np.ndarray = waveform.mean(axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported reference audio channel conversion: {current_channels} -> {target_channels}")
        return waveform.T.reshape(1, target_channels, -1).astype(np.float32, copy=False)

    def encode_reference_audio(self, reference_audio_path: str | Path) -> list[list[int]]:
        """调用 codec encode，把参考音频编码为逐帧的 RVQ 音频码。"""
        waveform = self._load_reference_audio(reference_audio_path)
        waveform_length = int(waveform.shape[-1])
        outputs = self.sessions["codec_encode"].run(
            None,
            {
                "waveform": waveform,
                "input_lengths": np.asarray([waveform_length], dtype=np.int32),
            },
        )
        output_names = [output.name for output in self.sessions["codec_encode"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        audio_codes = np.asarray(named_outputs["audio_codes"], dtype=np.int32)
        audio_code_lengths = np.asarray(named_outputs["audio_code_lengths"], dtype=np.int32)
        code_length = int(audio_code_lengths.reshape(-1)[0])
        num_quantizers = int(self.codec_meta["codec_config"]["num_quantizers"])
        return [
            [int(audio_codes[0, frame_index, quantizer_index]) for quantizer_index in range(num_quantizers)]
            for frame_index in range(code_length)
        ]

    def synthesize_single_chunk(self, *, text: str, prompt_audio_codes: list[list[int]]) -> dict[str, Any]:
        """合成单个文本块，并把生成的音频码整段解码为波形。"""
        text_token_ids = self.encode_text(text)
        request_rows = self.build_voice_clone_request_rows(prompt_audio_codes, text_token_ids)
        generated_frames = self.generate_audio_frames(request_rows)
        channel_arrays, _audio_length = self.decode_full_audio(generated_frames)
        return {
            "text": text,
            "generated_frames": generated_frames,
            "waveform": _merge_audio_channels(channel_arrays),
        }

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
    ) -> dict[str, Any]:
        """执行完整语音克隆：清洗文本、分块逐段生成、拼接停顿并保存 WAV。"""
        if not prompt_audio_path:
            raise ValueError("prompt_audio_path is required for voice cloning.")
        if max_new_frames is not None:
            self.manifest["generation_defaults"]["max_new_frames"] = int(max_new_frames)
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        prepared = self.prepare_synthesis_text(
            text=text,
            enable_normalize_tts_text=enable_normalize_tts_text,
        )
        prepared_text: str = str(prepared["text"])
        prompt_audio_codes: list[list[int]] = self.encode_reference_audio(prompt_audio_path)
        text_chunks: list[str] = self.split_voice_clone_text(prepared_text, max_tokens=int(voice_clone_max_text_tokens))
        sample_rate: int = int(self.codec_meta["codec_config"]["sample_rate"])
        channels: int = int(self.codec_meta["codec_config"]["channels"])

        all_waveforms: list[np.ndarray] = []
        all_generated_frames: list[list[int]] = []
        for chunk_index, chunk_text in enumerate(text_chunks):
            chunk_result: dict[str, Any] = self.synthesize_single_chunk(text=chunk_text, prompt_audio_codes=prompt_audio_codes)
            all_waveforms.append(np.asarray(chunk_result["waveform"], dtype=np.float32))
            all_generated_frames.extend(chunk_result["generated_frames"])
            if chunk_index < len(text_chunks) - 1:
                pause_samples: int = max(0, int(round(sample_rate * self.estimate_inter_chunk_pause_seconds(chunk_text))))
                if pause_samples > 0:
                    all_waveforms.append(np.zeros((pause_samples, channels), dtype=np.float32))

        waveform: np.ndarray = _concat_waveforms(all_waveforms)
        resolved_output_path: Path = Path(output_audio_path).expanduser().resolve()
        audio_path: Path = _write_waveform_to_wav(resolved_output_path, waveform, sample_rate)
        return {
            "audio_path": str(audio_path),
            "waveform": waveform,
            "sample_rate": sample_rate,
            "audio_token_ids": np.asarray(all_generated_frames, dtype=np.int32),
            "text_chunks": text_chunks,
            "prepared_text": prepared_text,
            "normalization_method": prepared["normalization_method"],
        }
