import base64
import binascii
import logging
import os

from rapidocr_onnxruntime import RapidOCR

from configs import read_setting
from plugin_mount import ModelInstance
from utils import MODEL_DIR

logger = logging.getLogger(__name__)


def _decode_data_uri(uri: str) -> bytes:
    if not isinstance(uri, str) or not uri:
        raise ValueError("OCR input must be base64 encoded image")
    payload = uri
    if uri.startswith("data:"):
        _, sep, payload = uri.partition(",")
        if not sep:
            raise ValueError("OCR input must be base64 encoded image")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("OCR input must be base64 encoded image") from None


def _format_result(result) -> str:
    if not result:
        return ""
    lines = []
    for item in result:
        if not item:
            continue
        # det + rec: [box, text, score]; rec only: [text, score]; det only: [box]
        if len(item) >= 3:
            text = item[1]
        elif len(item) == 2:
            text = item[0]
        else:
            text = item[0]
        if text:
            lines.append(str(text))
    return "\n".join(lines)


def _chunk(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


class RapidOcrInstance(ModelInstance):
    def __init__(self, model_info: dict, det_model_path: str, rec_model_path: str, thread: int):
        super().__init__(model_info)
        self.instance_type = "ocr"
        kwargs = {}
        if det_model_path:
            kwargs["det_model_path"] = det_model_path
        if rec_model_path:
            kwargs["rec_model_path"] = rec_model_path
        if thread and thread > 0:
            kwargs["intra_op_num_threads"] = thread
            kwargs["inter_op_num_threads"] = thread
        self.engine = RapidOCR(**kwargs)

    @property
    def model_name(self) -> str:
        return 'PP-OCRv6'

    @property
    def model_type(self) -> str:
        return 'ocr'

    @property
    def runtime(self) -> str:
        return 'onnxruntime'

    def recognize(self, uri: str, *, max_tokens: int = 1024):
        image_bytes = _decode_data_uri(uri)
        logger.info("OCR 识别开始 image_bytes=%s", len(image_bytes))
        result, _ = self.engine(image_bytes)
        lines = _format_result(result).split("\n") if result else []
        logger.info("OCR 识别完成 lines=%s", len(lines))
        for index, line in enumerate(lines):
            yield _chunk(line + ("\n" if index < len(lines) - 1 else ""))


def _find_model_path(model_info: dict, keyword: str) -> str | None:
    for model in model_info.get("models", []):
        folder = model.get("folder")
        for model_file in model.get("files", []):
            name = model_file.get("name", "")
            if folder and name and keyword in name.lower():
                return os.path.join(MODEL_DIR, folder, name)
    return None


def create_instance(model_info: dict):
    det_model_path = _find_model_path(model_info, "det")
    rec_model_path = _find_model_path(model_info, "rec")
    if not det_model_path or not os.path.isfile(det_model_path):
        raise FileNotFoundError(f"OCR det model not found: {det_model_path}")
    if not rec_model_path or not os.path.isfile(rec_model_path):
        raise FileNotFoundError(f"OCR rec model not found: {rec_model_path}")
    thread = read_setting("llama_params").get("n_threads", 2)
    logger.info("初始化 RapidOCR det=%s rec=%s thread=%s", det_model_path, rec_model_path, thread)
    return RapidOcrInstance(
        model_info=model_info,
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        thread=thread,
    )
