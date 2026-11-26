from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, Literal
import uuid

from configs import get_setting_models, read_setting
from task_queue import add_task, queue_status

api = APIRouter(prefix="/api/v1")

@api.get("/models")
async def get_models():
    return {
        "models": get_setting_models('all'),
        "model_list": read_setting('use_models'),
    }


class LLMRequest(BaseModel):
    type: Literal["llm"]
    model_type: Literal["chat", "ocr", "tts"]
    model_name: str
    input_type: Literal["text", "image"]
    user_input: str
    scene: Optional[str] = None
    think_mode: bool
class QueueRequest(BaseModel):
    type: Literal["queue"]
    model_type: Literal["chat", "ocr", "tts"]
    tid: Optional[str] = None

ClientRequest = LLMRequest | QueueRequest

ws_session: dict[str, list[dict]] = {}
def put_session_cache(tid: str, message: dict) -> None:
    if tid not in ws_session:
        ws_session[tid] = []
    ws_session[tid].append(message)
    if len(ws_session[tid]) > 6:
        ws_session[tid].pop(0)
def get_session_cache(tid: str) -> list[dict]:
    return ws_session.get(tid, [])

@api.websocket("/ws")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    tid = str(uuid.uuid4())
    try:
        while True:
            raw_data = await websocket.receive_json()
            switch = raw_data["type"]
            if switch == "llm":
                try:
                    req = LLMRequest.model_validate(raw_data)
                    if req.model_type == "chat":
                        put_session_cache(tid, {"role": "user", "content": req.user_input})
                        add_task(
                            tid, req.model_type, req.model_name, websocket, scene=req.scene, messages=get_session_cache(tid),
                            on_result=lambda content: put_session_cache(tid, {"role": "assistant", "content": content}),
                        )
                        continue
                    elif req.model_type == "ocr":
                        # no cache image input
                        add_task(
                            tid, req.model_type, req.model_name, websocket, scene=req.scene, messages=[{"role": "user", "content": req.user_input}],
                            on_result=lambda content: put_session_cache(tid, {"role": "assistant", "content": content}),
                        )
                        continue
                    elif req.model_type == "tts":
                        # 文本转语音，参考音频固定使用内置音色
                        add_task(
                            tid, req.model_type, req.model_name, websocket, scene=req.scene, messages=[{"role": "user", "content": req.user_input}],
                        )
                        continue
                except ValueError:
                    continue
            elif switch == "queue":
                try:
                    req = QueueRequest.model_validate(raw_data)
                except ValueError:
                    continue
                await websocket.send_json({
                    "type": "queue",
                    "tid": tid,
                    "model_type": req.model_type,
                    "content": queue_status(req.model_type, tid),
                })
            else:
                continue
    except WebSocketDisconnect:
        pass
    finally:
        if tid in ws_session:
            del ws_session[tid]
