import os
from configs import get_target_support_model
from modelscope_hub import HubApi, RepoType


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
hub_api = HubApi()

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
                if size and os.path.getsize(path) != size:
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
