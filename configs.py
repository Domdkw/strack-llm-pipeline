import json
import sys
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
default_setting = {
    "use_models": {
        "chat": [],
        "ocr": [],
        "tts": [],
    },
    "llama_params": {
        "temperature": 0.8,
        "n_threads": 0,
        "n_ctx": 4096,
        "n_gpu_layers": 0,
        "verbose": False,
    },
}
setting = {}
is_setting_updated = True
supports = {}

def write_setting(root_key: str, value: str|dict|list|bool|int|float|None):
    try:
        with open(os.path.join(ROOT_DIR, "setting.json"), "r+w", encoding="utf-8") as f:
            data = json.load(f)
            data[root_key] = value
            json.dump(data, f, ensure_ascii=False, indent=4)
            global is_setting_updated
            global setting
            is_setting_updated = True
            setting = data
            return True
    except Exception as e:
        print(e)
        print("Failed to write the setting.json.")
        sys.exit(1)

def read_setting(key: str):
    global setting
    global is_setting_updated
    if not is_setting_updated:
        return setting[key]
    try:
        with open(os.path.join(ROOT_DIR, "setting.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
            is_setting_updated = False
            setting = data
    except FileNotFoundError:
        with open(os.path.join(ROOT_DIR, "setting.json"), "w", encoding="utf-8") as f:
            json.dump(default_setting, f, ensure_ascii=False, indent=4)
            is_setting_updated = True
            setting = default_setting
        return default_setting[key]
    except Exception as e:
        print(e)
        print("Failed to read the setting.json.")
        sys.exit(1)
    else:
        if key in data:
            return data[key]
        else:
            return default_setting[key]

def read_supports():
    global supports
    if supports:
        return supports
    try:
        with open(os.path.join(ROOT_DIR, "supports.json"), "r", encoding="utf-8") as f:
            supports = json.load(f)
    except FileNotFoundError:
        print("Failed to read the supports.json.")
        sys.exit(1)
    return supports

def get_support_models(type: str):
    supports: dict = read_supports()
    if type != 'all':
        return supports[type]
    else:
        return supports['chat'] + supports['ocr'] + supports['tts']
def get_setting_models(type: str):
    setting = read_setting("use_models")
    if type != 'all':
        return setting[type]
    else:
        return setting['chat'] + setting['ocr'] + setting['tts']

def get_target_support_model(type: str, model_name: str):
    for block in get_support_models(type):
        if block['name'] == model_name:
            return block
    return None