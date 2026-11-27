from configs import get_target_support_model, read_setting
from utils import MODEL_DIR
from plugin_mount import get_pyEntrypoint_instance

import llama_runtime
import logging
import os

logger = logging.getLogger(__name__)

chat_completion = None
ocr_completions: dict[str, object] = {}
tts_completion = None

def build_llama_params():
    llama_params = read_setting("llama_params")
    temperature = llama_params["temperature"]
    n = {
        "n_ctx": llama_params["n_ctx"],
        "n_gpu_layers": llama_params["n_gpu_layers"],
        "n_threads": llama_params["n_threads"],
        "verbose": llama_params["verbose"],
    }
    return temperature, n

def get_type_file_path(model_info, type: str):
    for model_file in model_info["files"]:
        if model_file["type"] == type:
            return os.path.join(MODEL_DIR, model_info["folder"], model_file["name"])
    return None

def create_chat_model(model_name: str):
    global chat_completion
    if not chat_completion:
        model_info = get_target_support_model('chat', model_name)
        if not model_info or len(model_info["models"]) == 0:
            raise ValueError(f"Model {model_name} not found.")
        if model_info['adapter'] == "llama":
            gguf_path = get_type_file_path(model_info["models"][0], "gguf")
            temperature, n = build_llama_params()
            logger.info("加载 Chat 模型 %s gguf=%s", model_name, gguf_path)
            chat_completion = llama_runtime.ChatModel(gguf_path, temperature=temperature, n=n)
    return chat_completion

def create_ocr_model(model_name: str):
    if model_name in ocr_completions:
        return ocr_completions[model_name]
    model_info = get_target_support_model('ocr', model_name)
    if not model_info or len(model_info["models"]) == 0:
        raise ValueError(f"Model {model_name} not found.")
    ocr_completion = None
    if model_info['adapter'] == "llama":
        gguf_path = get_type_file_path(model_info["models"][0], "gguf")
        mmproj_path = get_type_file_path(model_info["models"][0], "mmproj")
        jinja_path = get_type_file_path(model_info["models"][0], "jinja")
        temperature, n = build_llama_params()
        logger.info("加载 OCR 模型 %s gguf=%s", model_name, gguf_path)
        ocr_completion = llama_runtime.OCRModel(gguf_path, mmproj_path, jinja_path, temperature=temperature, n=n)
    elif model_info['adapter'] == "python_entrypoint":
        ocr_completion = get_pyEntrypoint_instance(model_info)
    if ocr_completion is not None:
        ocr_completions[model_name] = ocr_completion
    return ocr_completion

def create_tts_model(model_name: str):
    global tts_completion
    if not tts_completion:
        model_info = get_target_support_model('tts', model_name)
        if not model_info or len(model_info["models"]) == 0:
            raise ValueError(f"Model {model_name} not found.")
        if model_info['adapter'] == "llama":
            pass
        if model_info['adapter'] == "python_entrypoint":
            tts_completion = get_pyEntrypoint_instance(model_info)   
    return tts_completion

def close_ocr_model(model_instance) -> None:
    """关闭 OCR 模型；若仍是缓存实例则同步清空缓存，避免后续任务复用已关闭实例。"""
    for name, cached in list(ocr_completions.items()):
        if cached is model_instance:
            del ocr_completions[name]
    model_instance.close()
    logger.info("gc collect ocr")