from gc import collect
import importlib.util
import os
import sys
from utils import PLUGIN_DIR
from abc import abstractmethod, ABC
from configs import get_target_support_model

model_instances = {}

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
    # 用唯一模块名，避免多个插件或主程序都叫 main 时相互覆盖
    module_name = f"strack_plugin_{entrypoint}"
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(path, "main.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    instance: ModelInstance = module.create_instance(model_info)
    model_name = instance.model_name
    if model_name != info_name:
        raise ValueError(f"Model {info_name} expect {model_name}")
    model_instances[model_name] = instance
    return instance

def get_model_instance(model_name: str) -> ModelInstance|None:
    return model_instances.get(model_name, None)

def create_model_instance(type: str, model_name: str) -> ModelInstance|None:
    model_info = get_target_support_model(type, model_name)
    if not model_info:
        return None
    return get_pyEntrypoint_instance(model_info)
