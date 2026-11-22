from __future__ import annotations
from typing import TYPE_CHECKING
import asyncio
from dataclasses import dataclass
from contextlib import asynccontextmanager
from model_bootstrap import create_chat_model, create_ocr_model
from configs import get_setting_models
import base64
import binascii

if TYPE_CHECKING:
    from llama_runtime import ModelInstance
    from fastapi import WebSocket

@dataclass
class Task:
    tid: str
    model_instance: ModelInstance
    ws_instance: WebSocket
    messages: list[dict]
    status: str = "pending"  # pending, running, done, failed
    result: str = ""  # 模型完整输出
    on_result: callable | None = None  # 输出收集回调，参数为完整文本

chat_queue = asyncio.Queue[Task](maxsize=100)
ocr_queue = asyncio.Queue[Task](maxsize=10)

def _detect_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None

def parse_ocr_image(content: str) -> tuple[bytes, str]:
    """解析 OCR 输入，返回 (图片字节, mime)。支持 data URI 与裸 Base64。"""
    if not isinstance(content, str) or not content:
        raise ValueError("OCR input must be base64 encoded image")
    payload = content
    declared_mime = None
    if content.startswith("data:"):
        header, sep, payload = content.partition(",")
        if not sep or not header.startswith("data:image/") or ";base64" not in header:
            raise ValueError("OCR input must be base64 encoded image")
        declared_mime = header[len("data:"):-len(";base64")]
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("OCR input must be base64 encoded image") from None
    mime = _detect_image_mime(raw)
    if mime is None:
        raise ValueError("OCR input must be base64 encoded image")
    if declared_mime and declared_mime != mime:
        raise ValueError(f"OCR image mime mismatch: declared {declared_mime}, actual {mime}")
    return raw, mime

def verify_ocr_input(message: dict) -> None:
    parse_ocr_image(message.get("content"))

def queue_status(model_type: str, tid: str | None = None) -> dict:
    if model_type == "chat":
        queue = chat_queue
    elif model_type == "ocr":
        queue = ocr_queue
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    pending_tids = [task.tid for task in queue._queue]
    status = {
        "size": queue.qsize(),
        "maxsize": queue.maxsize,
        "position": pending_tids.index(tid) + 1 if tid in pending_tids else None,
    }
    return status

def add_task(tid: str, model_type: str, model_name: str, ws: WebSocket, messages: list[dict], on_result: callable | None = None) -> None:
    used_models = get_setting_models(model_type)
    if model_name not in used_models:
        raise ValueError(f"Model {model_name} not found in {model_type} models")
    if model_type == "chat":
        model_instance = create_chat_model(model_name)
        chat_queue.put_nowait(Task(
            tid=tid,
            model_instance=model_instance,
            ws_instance=ws,
            messages=list[dict](messages),
            on_result=on_result,
        ))
    elif model_type == "ocr":
        if not messages:
            raise ValueError("OCR input must be base64 encoded image")
        verify_ocr_input(messages[0])
        model_instance = create_ocr_model(model_name)
        content = messages[0]["content"]
        if not content.startswith("data:"):
            _, mime = parse_ocr_image(content)
            content = f"data:{mime};base64,{content}"
        ocr_queue.put_nowait(Task(
            tid=tid,
            model_instance=model_instance,
            ws_instance=ws,
            messages=[{"role": "user", "content": content}],
        ))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def forEach_chunk(response, callback: callable):
    parts = []
    for chunk in response:
        delta = chunk["choices"][0].get("delta", {})
        text = delta.get("content")
        if text:
            parts.append(text)
            callback(text)
    return "".join(parts)

async def _stream_task(task: Task, invoke: callable):
    """在线程中执行阻塞的模型调用，并把 chunk 逐个发回 WebSocket。"""
    loop = asyncio.get_running_loop()
    chunks: asyncio.Queue[str | None] = asyncio.Queue()

    def emit(text: str):
        loop.call_soon_threadsafe(chunks.put_nowait, text)

    def run():
        try:
            return forEach_chunk(invoke(), emit)
        finally:
            loop.call_soon_threadsafe(chunks.put_nowait, None)

    work = asyncio.create_task(asyncio.to_thread(run))
    try:
        while True:
            text = await chunks.get()
            if text is None:
                break
            await task.ws_instance.send_json({
                "type": "llm",
                "tid": task.tid,
                "model_type": task.model_instance.instance_type,
                "message_len": len(task.messages),
                "content": { "chunk": text },
            })
    finally:
        task.result = await work
    return task.result

async def chat_worker():
    while True:
        task = await chat_queue.get()
        try:
            task.status = "running"
            # 调用模型（阻塞操作放到线程，避免卡住事件循环）
            content = await _stream_task(task, lambda: task.model_instance.chat(task.messages, max_tokens=1024))
            task.result = content
            task.status = "done"
            if task.on_result:
                task.on_result(content)
        except Exception:
            task.status = "failed"  # 避免单个任务异常导致 worker 退出
        finally:
            chat_queue.task_done()

async def ocr_worker():
    while True:
        task = await ocr_queue.get()
        try:
            task.status = "running"
            content = await _stream_task(task, lambda: task.model_instance.recognize(task.messages[0]["content"], max_tokens=1024))
            task.result = content
            task.status = "done"
        except Exception:
            task.status = "failed"
        finally:
            ocr_queue.task_done()
