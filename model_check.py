import os
import shutil
from configs import get_target_support_model, get_setting_models
from utils import MODEL_DIR as MODELS_DIR
from modelscope_hub import HubApi, RepoType


hub_api = HubApi()

def scan_models():
    used_models = get_setting_models('all')
    if not used_models:
        print("[scan] use_models: skip scan.")
        return
    print(f"[scan] 检查模型: {', '.join(used_models)}")
    for model_name in used_models:
        if not has_model(model_name):
            print(f"[scan] {model_name} 未安装，开始下载。")
            download_model(model_name)

def has_model(model_name: str, model_type: str = 'all') -> bool:
    target_model = get_target_support_model(model_type, model_name)
    if target_model:
        for model in target_model.get('models', []):
            folder = model['folder']
            model_files = model.get('files', [])
            if not model_files:
                continue
            for model_file in model_files:
                file_name = model_file.get('name')
                path =  os.path.join(MODELS_DIR, folder, file_name)
                size = model_file.get('size', 0)
                print(f"[has_model] Check {model_name}: {file_name}")
                if not os.path.isfile(path):
                    return False
                if size and size > 0 and os.path.getsize(path) != size:
                    return False
        return True
    return False

def download_model(model_name: str) -> bool:
    target_model = get_target_support_model('all', model_name)
    if target_model:
        if has_model(model_name):
            print(f"[dl_model] Model {model_name} has been installed.")
            return True
        for model in target_model.get('models', []):
            model_id = model.get('modelscope_id')
            folder = model.get('folder')
            if not model_id or not folder:
                print(f"[dl_model] Invalid config for {model_name}: {model}")
                return False
            local_dir = os.path.join(MODELS_DIR, folder)
            os.makedirs(local_dir, exist_ok=True)
            # 模型仓库内路径与实际本地文件名不同时，按 modelscope_path 单独下载后复制到目标位置
            modelscope_path = model.get('modelscope_path')
            if modelscope_path:
                for model_file in model.get('files', []):
                    file_name = model_file.get('name')
                    if not file_name:
                        continue
                    target_path = os.path.join(local_dir, file_name)
                    size = model_file.get('size', 0)
                    if os.path.isfile(target_path) and (not size or size <= 0 or os.path.getsize(target_path) == size):
                        continue
                    print(f"[dl_model] Downloading {model_id}/{modelscope_path} -> {target_path}")
                    try:
                        downloaded = hub_api.download_file(
                            repo_id=model_id,
                            repo_type=RepoType.MODEL,
                            file_path=modelscope_path,
                        )
                        shutil.copyfile(downloaded, target_path)
                    except Exception as e:
                        print(f"[dl_model] Download failed: {e}")
                        return False
                continue
            patterns = [f['name'] for f in model.get('files', []) if f.get('name')]
            print(f"[dl_model] Downloading {model_id} -> {local_dir}")
            try:
                hub_api.download_repo(
                    repo_id=model_id,
                    repo_type=RepoType.MODEL,
                    local_dir=local_dir,
                    allow_patterns=patterns or None,
                )
            except Exception as e:
                print(f"[dl_model] Download failed: {e}")
                return False
        installed = has_model(model_name)
        print(f"[dl_model] Model {model_name} "
              f"{'downloaded.' if installed else 'download incomplete.'}")
        return installed
    print(f"Model {model_name} is not supported.")
    return False
