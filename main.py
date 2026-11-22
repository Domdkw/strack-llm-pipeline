from fastapi import FastAPI
import asyncio
import uvicorn
from contextlib import asynccontextmanager

from model_check import has_model, download_model
from configs import get_setting_models
from llm_service import api
from task_queue import chat_worker, ocr_worker

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    workers = [asyncio.create_task(chat_worker()), asyncio.create_task(ocr_worker())]
    try:
        yield
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

app = FastAPI(title="Strack LLM Pipeline", lifespan=lifespan)
app.include_router(api)
@app.middleware("http")
async def add_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["x-strack-llm-pipeline-version"] = "1.0.0"
    response.headers["min-strack-version"] = "1.1.0"
    return response

def main():
    scan_models()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()