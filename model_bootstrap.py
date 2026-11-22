from configs import get_target_support_model, ROOT_DIR, read_setting
from llama_runtime import ChatModel, OCRModel
import os

chat_completion = None
ocr_completion = None

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
            return os.path.join(ROOT_DIR, "models", model_info["folder"], model_file["name"])
    return None

def create_chat_model(model_name: str):
    global chat_completion
    if not chat_completion:
        model_info = get_target_support_model('chat', model_name)["models"][0]
        if not model_info:
            raise ValueError(f"Model {model_name} not found.")
        gguf_path = get_type_file_path(model_info, "gguf")
        temperature, n = build_llama_params()
        chat_completion = ChatModel(gguf_path, temperature=temperature, n=n)
    return chat_completion

def create_ocr_model(model_name: str):
    global ocr_completion
    if not ocr_completion:
        model_info = get_target_support_model('ocr', model_name)["models"][0]
        if not model_info:
            raise ValueError(f"Model {model_name} not found.")
        gguf_path = get_type_file_path(model_info, "gguf")
        mmproj_path = get_type_file_path(model_info, "mmproj")
        jinja_path = get_type_file_path(model_info, "jinja")
        temperature, n = build_llama_params()
        ocr_completion = OCRModel(gguf_path, mmproj_path, jinja_path, temperature=temperature, n=n)
    return ocr_completion

def close_ocr_model(model_instance) -> None:
    """关闭 OCR 模型；若仍是缓存实例则同步清空缓存，避免后续任务复用已关闭实例。"""
    global ocr_completion
    if ocr_completion is model_instance:
        ocr_completion = None
    model_instance.close()