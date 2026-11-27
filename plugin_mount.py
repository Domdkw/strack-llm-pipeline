from gc import collect
import importlib.util
import logging
import os
import sys
import time
from utils import PLUGIN_DIR, PUBLIC_TEMP_DIR
from abc import abstractmethod, ABC
from configs import get_target_support_model

model_instances = {}
logger = logging.getLogger(__name__)

class ModelInstance(ABC):
    @abstractmethod
    def __init__(self, model_info: dict):
        pass
    @property
    @abstractmethod
    def model_name(self) -> str:
        return self._model_name
    @property
    @abstractmethod
    def model_type(self) -> str:
        return self._model_type
    @property
    @abstractmethod
    def runtime(self) -> str:
        return self._runtime
    def close(self):
        collect()


def get_pyEntrypoint_instance(model_info) -> ModelInstance:
    info_name = model_info["name"]
    if info_name in model_instances:
        return model_instances[info_name]
    entrypoint = model_info["entrypoint"]
    path = os.path.join(PLUGIN_DIR, entrypoint)
    # 插件内部的同级模块（如 onnx_tts_runtime）需要能按绝对名导入
    if path not in sys.path:
        sys.path.insert(0, path)
    # 用唯一模块名，避免main相互覆盖
    module_name = f"strack_plugin_{entrypoint}"
    logger.info("加载插件 entrypoint=%s model=%s", entrypoint, info_name)
    try:
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(path, "main.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        instance: ModelInstance = module.create_instance(model_info)
    except Exception:
        logger.exception("插件加载失败 entrypoint=%s model=%s", entrypoint, info_name)
        raise
    model_name = instance.model_name
    if model_name != info_name:
        raise ValueError(f"Model {info_name} expect {model_name}")
    model_instances[model_name] = instance
    logger.info("插件加载完成 model=%s runtime=%s", model_name, instance.runtime)
    return instance

def get_model_instance(model_name: str) -> ModelInstance|None:
    return model_instances.get(model_name, None)

def create_model_instance(type: str, model_name: str) -> ModelInstance|None:
    model_info = get_target_support_model(type, model_name)
    if not model_info:
        return None
    return get_pyEntrypoint_instance(model_info)

def register_temp_file(file_name: str, expires: int = 60 * 5, public: bool = True) -> str:
    """ 注册一个临时文件，过期时间为 expires 秒，返回文件路径 """
    name = f"0_{file_name}"
    if expires > 0:
        timestamp: int = int(time.time()) + expires
        name = f"{timestamp}_{file_name}"
    if public:
        os.makedirs(PUBLIC_TEMP_DIR, exist_ok=True)
        path = os.path.join(PUBLIC_TEMP_DIR, name)
    else:
        os.makedirs(os.path.join(PUBLIC_TEMP_DIR, "private"), exist_ok=True)
        path = os.path.join(PUBLIC_TEMP_DIR, "private", name)
    return path