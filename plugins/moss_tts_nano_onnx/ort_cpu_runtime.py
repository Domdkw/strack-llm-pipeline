from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

MANIFEST_CANDIDATE_RELATIVE_PATHS = (
    "browser_poc_manifest.json",
    "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
    "MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json",
)
MODEL_DIR_ALIAS_MAP = {
    "MOSS-TTS-Nano-ONNX-CPU": "MOSS-TTS-Nano-100M-ONNX",
    "MOSS-Audio-Tokenizer-Nano-ONNX-CPU": "MOSS-Audio-Tokenizer-Nano-ONNX",
}


def _flatten3d_int32(nested: list[list[list[int]]]) -> tuple[np.ndarray, list[int]]:
    """把三维嵌套列表展平为一维 int32 数组，并返回原始维度。"""
    dim0 = len(nested)
    dim1 = len(nested[0])
    dim2 = len(nested[0][0])
    data = np.zeros((dim0 * dim1 * dim2,), dtype=np.int32)
    offset = 0
    for i in range(dim0):
        for j in range(dim1):
            for k in range(dim2):
                data[offset] = int(nested[i][j][k])
                offset += 1
    return data, [dim0, dim1, dim2]


def _flatten2d_int32(nested: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    """把二维嵌套列表展平为一维 int32 数组，并返回原始维度。"""
    dim0 = len(nested)
    dim1 = len(nested[0])
    data = np.zeros((dim0 * dim1,), dtype=np.int32)
    offset = 0
    for i in range(dim0):
        for j in range(dim1):
            data[offset] = int(nested[i][j])
            offset += 1
    return data, [dim0, dim1]


def _slice_channel_major_audio(audio: np.ndarray, start_sample: int = 0, end_sample: int | None = None) -> list[np.ndarray]:
    """从 (1, channels, samples) 音频张量截取采样区间，按声道返回数组列表。"""
    if audio.ndim != 3 or audio.shape[0] != 1:
        raise ValueError(f"Unexpected audio tensor shape: {audio.shape}")
    channels = int(audio.shape[1])
    total_samples = int(audio.shape[2])
    start = max(0, int(start_sample))
    end = total_samples if end_sample is None else max(start, min(int(end_sample), total_samples))
    return [audio[0, channel_index, start:end].astype(np.float32, copy=False) for channel_index in range(channels)]


def _extract_last_hidden(hidden_states: np.ndarray) -> np.ndarray:
    """取 hidden states 的最后一帧，作为下一步 decode 使用的全局隐状态。"""
    if hidden_states.ndim == 2:
        return hidden_states.astype(np.float32, copy=False)
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError(f"Unexpected global_hidden shape: {hidden_states.shape}")
    return hidden_states[:, -1, :].astype(np.float32, copy=False)


class OrtCpuRuntime:
    """ONNX Runtime CPU 推理内核，只走 fixed 采样路径。"""

    def __init__(
        self,
        thread_count: int = 4,
        max_new_frames: int | None = None,
        tts_model_dir: str | Path | None = None,
        codec_model_dir: str | Path | None = None,
    ) -> None:
        """读取 manifest 与模型元数据，并创建 ONNX 会话。

        tts_model_dir / codec_model_dir 可手动指定 TTS 与 codec 两个模型的目录，
        覆盖 manifest 中的相对路径。
        """
        self.thread_count = max(1, int(thread_count))
        self.tts_meta_path = Path(tts_model_dir) / "tts_browser_onnx_meta.json"
        self.tts_meta = json.loads(self.tts_meta_path.read_text(encoding="utf-8"))
        self.manifest_path = Path(tts_model_dir) / "browser_poc_manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.codec_meta_path = Path(codec_model_dir) / "codec_browser_onnx_meta.json"
        self.codec_meta = json.loads(self.codec_meta_path.read_text(encoding="utf-8"))
        self.tokenizer_path = Path(tts_model_dir) / "tokenizer.model"
        if max_new_frames is not None:
            self.manifest["generation_defaults"]["max_new_frames"] = int(max_new_frames)
        self.rng = np.random.default_rng(1234)
        self.sessions = self._create_sessions()

    def _session(self, path_value: Path) -> ort.InferenceSession:
        """用指定线程数创建 CPU 版 ONNX InferenceSession。"""
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = self.thread_count
        options.inter_op_num_threads = 1
        return ort.InferenceSession(str(path_value), sess_options=options, providers=["CPUExecutionProvider"])

    def _create_sessions(self) -> dict[str, ort.InferenceSession]:
        """创建 fixed 采样所需的 TTS 与 codec ONNX 会话。"""
        tts_dir = self.tts_meta_path.parent
        codec_dir = self.codec_meta_path.parent
        tts_files = self.tts_meta["files"]
        required_frame_model = tts_files.get("local_fixed_sampled_frame")
        if not required_frame_model:
            raise RuntimeError(
                "The ONNX model does not provide local_fixed_sampled_frame; this simplified runtime only supports fixed sampling."
            )
        return {
            "prefill": self._session(tts_dir / tts_files["prefill"]),
            "decode": self._session(tts_dir / tts_files["decode_step"]),
            "local_fixed_sampled_frame": self._session(tts_dir / required_frame_model),
            "codec_encode": self._session(codec_dir / self.codec_meta["files"]["encode"]),
            "codec_decode": self._session(codec_dir / self.codec_meta["files"]["decode_full"]),
        }

    def build_text_rows(self, token_ids: list[int]) -> list[list[int]]:
        """把文本 token 扩展为“文本 token + 音频占位”的请求行。"""
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        rows: list[list[int]] = []
        for token_id in token_ids:
            row = [int(self.manifest["tts_config"]["audio_pad_token_id"])] * row_width
            row[0] = int(token_id)
            rows.append(row)
        return rows

    def build_audio_prefix_rows(self, prompt_audio_codes: list[list[int]]) -> list[list[int]]:
        """把参考音频码扩展为以 audio_user_slot 开头的请求行。"""
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        slot_token_id = int(self.manifest["tts_config"]["audio_user_slot_token_id"])
        rows: list[list[int]] = []
        for code_row in prompt_audio_codes:
            row = [int(self.manifest["tts_config"]["audio_pad_token_id"])] * row_width
            row[0] = slot_token_id
            for index in range(min(len(code_row), int(self.manifest["tts_config"]["n_vq"]))):
                row[index + 1] = int(code_row[index])
            rows.append(row)
        return rows

    def build_voice_clone_request_rows(self, prompt_audio_codes: list[list[int]], text_token_ids: list[int]) -> dict[str, list[list[int]]]:
        """拼装音色克隆请求：用户前缀 + 参考音频码 + 待合成文本 + 助手前缀。"""
        prefix_text_token_ids = [
            *self.manifest["prompt_templates"]["user_prompt_prefix_token_ids"],
            int(self.manifest["tts_config"]["audio_start_token_id"]),
        ]
        suffix_text_token_ids = [
            int(self.manifest["tts_config"]["audio_end_token_id"]),
            *self.manifest["prompt_templates"]["user_prompt_after_reference_token_ids"],
            *text_token_ids,
            *self.manifest["prompt_templates"]["assistant_prompt_prefix_token_ids"],
            int(self.manifest["tts_config"]["audio_start_token_id"]),
        ]
        rows = [
            *self.build_text_rows(prefix_text_token_ids),
            *self.build_audio_prefix_rows(prompt_audio_codes),
            *self.build_text_rows(suffix_text_token_ids),
        ]
        return {
            "inputIds": rows,
            "attentionMask": [[1 for _ in rows]],
        }

    def run_local_fixed_sampled_frame(
        self,
        global_hidden: np.ndarray,
        *,
        previous_token_sets_by_channel: list[set[int]],
    ) -> tuple[bool, list[int]]:
        """调用 local_fixed_sampled_frame 采样一帧音频码，返回是否继续与码值列表。"""
        audio_codebook_size = int(self.tts_meta["model_config"]["audio_codebook_sizes"][0])
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        repetition_seen_mask = np.zeros((1, n_vq, audio_codebook_size), dtype=np.int32)
        for channel_index, token_ids in enumerate(previous_token_sets_by_channel):
            for token_id in token_ids:
                if 0 <= token_id < audio_codebook_size:
                    repetition_seen_mask[0, channel_index, token_id] = 1
        assistant_random_u = np.asarray([min(0.99999994, max(0.0, float(self.rng.random())))], dtype=np.float32)
        audio_random_u = np.asarray(
            [[min(0.99999994, max(0.0, float(self.rng.random()))) for _ in range(n_vq)]],
            dtype=np.float32,
        )
        outputs = self.sessions["local_fixed_sampled_frame"].run(
            None,
            {
                "global_hidden": global_hidden.astype(np.float32, copy=False),
                "repetition_seen_mask": repetition_seen_mask,
                "assistant_random_u": assistant_random_u,
                "audio_random_u": audio_random_u,
            },
        )
        output_names = [output.name for output in self.sessions["local_fixed_sampled_frame"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        frame_token_ids = np.asarray(named_outputs["frame_token_ids"]).reshape(-1).astype(np.int32, copy=False).tolist()
        should_continue = bool(int(np.asarray(named_outputs["should_continue"]).reshape(-1)[0]))
        return should_continue, [int(item) for item in frame_token_ids]

    def decode_full_audio(self, generated_frames: list[list[int]]) -> tuple[list[np.ndarray], int]:
        """把完整音频码序列整段解码为按声道分开的波形及有效采样数。"""
        if not generated_frames:
            return [], 0
        audio_codes, dims = _flatten3d_int32([generated_frames])
        outputs = self.sessions["codec_decode"].run(
            None,
            {
                "audio_codes": audio_codes.reshape(dims),
                "audio_code_lengths": np.asarray([len(generated_frames)], dtype=np.int32),
            },
        )
        output_names = [output.name for output in self.sessions["codec_decode"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        audio_length = int(named_outputs["audio_lengths"].reshape(-1)[0])
        return _slice_channel_major_audio(named_outputs["audio"], 0, audio_length), audio_length

    def generate_audio_frames(self, request_rows: dict[str, list[list[int]]]) -> list[list[int]]:
        """执行 prefill + 逐帧采样 + KV 缓存 decode，生成整段音频码。"""
        defaults = self.manifest["generation_defaults"]
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        prefill_ids, prefill_dims = _flatten3d_int32([request_rows["inputIds"]])
        prefill_mask, prefill_mask_dims = _flatten2d_int32(request_rows["attentionMask"])
        outputs = self.sessions["prefill"].run(
            None,
            {
                "input_ids": prefill_ids.reshape(prefill_dims),
                "attention_mask": prefill_mask.reshape(prefill_mask_dims),
            },
        )
        output_names = [output.name for output in self.sessions["prefill"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        global_hidden = _extract_last_hidden(named_outputs["global_hidden"])
        past_valid_length = sum(int(item) for item in request_rows["attentionMask"][0])
        past_by_name = {
            output_name.replace("present_", "past_"): named_outputs[output_name]
            for output_name in self.tts_meta["onnx"]["prefill_output_names"][1:]
        }
        generated_frames: list[list[int]] = []
        previous_token_sets_by_channel = [set() for _ in range(int(self.manifest["tts_config"]["n_vq"]))]

        for _ in range(int(defaults["max_new_frames"])):
            should_continue, frame = self.run_local_fixed_sampled_frame(
                global_hidden,
                previous_token_sets_by_channel=previous_token_sets_by_channel,
            )
            if not should_continue:
                break
            for channel_index, sampled_token in enumerate(frame):
                previous_token_sets_by_channel[channel_index].add(sampled_token)
            generated_frames.append(frame)

            next_row = np.full((1, 1, row_width), int(self.manifest["tts_config"]["audio_pad_token_id"]), dtype=np.int32)
            next_row[0, 0, 0] = int(self.manifest["tts_config"]["audio_assistant_slot_token_id"])
            for index, token in enumerate(frame):
                next_row[0, 0, index + 1] = int(token)
            decode_feeds: dict[str, np.ndarray] = {
                "input_ids": next_row,
                "past_valid_lengths": np.asarray([past_valid_length], dtype=np.int32),
            }
            for input_name in self.tts_meta["onnx"]["decode_input_names"][2:]:
                decode_feeds[input_name] = past_by_name[input_name]
            decode_outputs = self.sessions["decode"].run(None, decode_feeds)
            decode_output_names = [output.name for output in self.sessions["decode"].get_outputs()]
            named_decode_outputs = dict(zip(decode_output_names, decode_outputs, strict=True))
            global_hidden = _extract_last_hidden(named_decode_outputs["global_hidden"])
            past_valid_length += 1
            past_by_name = {
                output_name.replace("present_", "past_"): named_decode_outputs[output_name]
                for output_name in self.tts_meta["onnx"]["decode_output_names"][1:]
            }
        return generated_frames


__all__ = ["OrtCpuRuntime"]
