"""命令行交互测试脚本：测试 llama_runtime.py 中的 ChatModel / OCRModel。

用法:
    # 纯文本多轮对话（经 model_bootstrap 加载 Qwen3-0.6B）
    python test/test_models.py

    # OCR 识别图片（默认自动使用模型目录下的 chat_template.jinja）
    python test/test_models.py --image path/to/image.png

    # 指定 jinja 模板
    python test/test_models.py --image path/to/image.png \
        --jinja models/paddleocr-vl-1.6/ws_template.jinja
"""

import argparse
import base64
import importlib.util
import mimetypes
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_bootstrap import create_chat_model, create_ocr_model

CHAT_MODEL_NAME = "Qwen3-0.6B"
OCR_MODEL = "PaddleOCR-VL-1.6"


def load_runtime():
    """按文件路径动态加载 llama_runtime.py。"""
    spec = importlib.util.spec_from_file_location("llama_runtime", ROOT / "llama_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_to_data_uri(path: pathlib.Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def stream_text(response) -> str:
    parts = []
    for chunk in response:
        delta = chunk["choices"][0].get("delta", {})
        text = delta.get("content")
        if text:
            parts.append(text)
            print(text, end="", flush=True)
    print()
    return "".join(parts)


def build_llama_params(args) -> dict:
    return {
        "n_threads": args.n_threads,
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "verbose": args.verbose,
    }


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run_ocr(runtime, args) -> None:
    image_path = pathlib.Path(args.image)
    if not image_path.is_file():
        sys.exit(f"图片不存在: {image_path}")

    log(f"[OCR] 加载模型: {OCR_MODEL}")
    model = create_ocr_model(OCR_MODEL)
    log(f"[OCR] chat template: {args.jinja or 'GGUF 内置'}")
    log(f"[OCR] 模型加载完成，n_ctx={args.n_ctx}，开始识别: {image_path}")
    log("[OCR] 首次识别需要加载视觉投影并编码图片，CPU 上可能需要较长时间，请耐心等待...\n")
    uri = image_to_data_uri(image_path)
    start = time.perf_counter()
    first = True
    try:
        for chunk in model.recognize(uri, max_tokens=args.max_tokens):
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content")
            if text:
                if first:
                    log(f"[OCR] 首个 token 耗时 {time.perf_counter() - start:.1f}s")
                    first = False
                print(text, end="", flush=True)
        print(f"\n[OCR] 完成，总耗时 {time.perf_counter() - start:.1f}s")
    finally:
        # 在解释器退出前显式释放，避免 __del__ 在 llama_cpp 模块已销毁后执行 mtmd_free
        model.close()


def run_chat(args) -> None:
    print(f"[CHAT] 从 model_bootstrap 加载: {CHAT_MODEL_NAME}")
    model = create_chat_model(CHAT_MODEL_NAME)
    print("[CHAT] 进入对话，输入 exit / quit 退出，clear 清空上下文。\n")

    messages = []
    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() == "clear":
            messages.clear()
            print("[CHAT] 上下文已清空。")
            continue

        messages.append({"role": "user", "content": user_input})
        print("<<< ", end="", flush=True)
        try:
            reply = stream_text(model.chat(messages, max_tokens=args.max_tokens))
            messages.append({"role": "assistant", "content": reply})
        except KeyboardInterrupt:
            print("\n[CHAT] 已中断本次生成。")
            messages.pop()


def parse_args():
    parser = argparse.ArgumentParser(description="llama_runtime.py 命令行交互测试")
    parser.add_argument("--model", type=pathlib.Path, help="OCR GGUF 模型路径")
    parser.add_argument("--mmproj", type=pathlib.Path, help="OCR 多模态投影 mmproj 路径")
    parser.add_argument("--jinja", type=pathlib.Path, help="外部 chat template jinja 路径")
    parser.add_argument("--image", type=str, help="OCR 图片路径，指定后进入 OCR 模式")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--n-ctx", type=int, default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="输出 llama.cpp 日志，便于定位卡住的位置")
    args = parser.parse_args()
    args.n_threads = args.n_threads or max(1, (os.cpu_count() or 4) - 1)

    if args.image:
        args.model = args.model or OCR_MODEL
        args.mmproj = args.mmproj or None
        args.jinja = args.jinja or None
        # OCR 推荐贪心解码，且图片会占用较多上下文
        args.temperature = 0.0 if args.temperature is None else args.temperature
        args.n_ctx = 4096 if args.n_ctx is None else args.n_ctx
    return args


def main():
    args = parse_args()
    if args.image:
        run_ocr(load_runtime(), args)
    else:
        run_chat(args)


if __name__ == "__main__":
    main()
