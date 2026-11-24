import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
PLUGIN_DIR = os.path.join(ROOT_DIR, "plugins")

def create_web_dir():
    if not os.path.exists(os.path.join(ROOT_DIR, "public")):
        os.makedirs(os.path.join(ROOT_DIR, "public"))
    if not os.path.exists(os.path.join(ROOT_DIR, "app")):
        os.makedirs(os.path.join(ROOT_DIR, "app"))
