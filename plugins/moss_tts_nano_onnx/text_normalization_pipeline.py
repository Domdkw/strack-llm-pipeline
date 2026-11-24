from __future__ import annotations

import logging
from typing import Any

from tts_robust_normalizer_single_script import normalize_tts_text


def prepare_tts_request_texts(
    *,
    text: str,
    enable_normalize_tts_text: bool = True,
) -> dict[str, Any]:
    """对 TTS 输入做鲁棒性字符清洗，不做语义数字/单位展开。"""
    raw_text = str(text or "")
    final_text = normalize_tts_text(raw_text) if enable_normalize_tts_text else raw_text
    if final_text != raw_text:
        logging.info(
            "normalized text chars_before=%d chars_after=%d stage=robust",
            len(raw_text),
            len(final_text),
        )
    return {
        "text": final_text,
        "normalization_method": "robust" if enable_normalize_tts_text else "none",
    }
