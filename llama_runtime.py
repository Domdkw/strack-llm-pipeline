from llama_cpp import Llama
from llama_cpp.llama_chat_format import MTMDChatHandler
import pathlib

class ChatModel:
    def __init__(self, model_path: str, temperature=0.8, n: dict = None):
        n_params = n or {}
        self.temperature: float = temperature
        self.instance_type: str = "chat"
        self.llm = Llama(
            model_path=str(model_path),
            n_threads=n_params.get("n_threads", 4),
            n_ctx=n_params.get("n_ctx", 512),
            use_mmap=True,
            verbose=n_params.get("verbose", False),
            n_gpu_layers=n_params.get("n_gpu_layers", 0),
        )
    
    def chat(self, messages: list, max_tokens: int = 1024):
        return self.llm.create_chat_completion(
            messages=[{"role": "system", "content": "你是一个专业的助手，除非用户明确需求，则默认使用中文回答。"}] + messages,
            stream=True,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )

    def close(self):
        self.llm.close()

class JinjaMTMDChatHandler(MTMDChatHandler):
    """MTMDChatHandler，支持用外部 jinja 模板覆盖 GGUF 内置的 chat template。"""

    def __init__(self, clip_model_path: str, chat_template: str = None, **kwargs):
        super().__init__(clip_model_path=clip_model_path, **kwargs)
        self.chat_template = chat_template

    def _get_chat_template(self, llama_model) -> str:
        if self.chat_template:
            return self.chat_template
        return super()._get_chat_template(llama_model)


class OCRModel: 
    def __init__(self, gguf_path: str, mmproj_path: str, jinja_path: str = None, temperature=0.8, n: dict = None):
        n_params = n or {}
        self.temperature: float = temperature
        self.instance_type: str = "ocr"
        chat_template = (
            pathlib.Path(jinja_path).read_text(encoding="utf-8") if jinja_path else None
        )
        self.llm = Llama(
            model_path=str(gguf_path),
            chat_handler=JinjaMTMDChatHandler(
                clip_model_path=str(mmproj_path),
                chat_template=chat_template,
                verbose=n_params.get("verbose", False),
                use_gpu=n_params.get("n_gpu_layers", 0) != 0,
            ),
            n_threads=n_params.get("n_threads", 4),
            n_ctx=n_params.get("n_ctx", 512),
            use_mmap=True,
            verbose=n_params.get("verbose", False),
            n_gpu_layers=n_params.get("n_gpu_layers", 0),
        )
    
    def recognize(self, uri: str, max_tokens: int = 1024):
        return self.llm.create_chat_completion(
            messages=[{"role": "user", "content": [
                    {"type": "text", "text": "OCR:"}, # OCR 模式
                    {"type": "image_url", "image_url": {"url": uri}} # 图片
            ]}],
            stream=True,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )

    def close(self):
        self.llm.close()

ModelInstance = ChatModel | OCRModel