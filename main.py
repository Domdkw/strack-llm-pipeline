from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager

from model_check import scan_models
from utils import create_web_dir
from llm_service import api
from task_queue import chat_worker, ocr_worker, tts_worker

STRACK_LLM_PIPELINE_VERSION = "1.0.0"
STRACK_MIN_VERSION = "1.1.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    workers = [
        asyncio.create_task(chat_worker()),
        asyncio.create_task(ocr_worker()),
        asyncio.create_task(tts_worker()),
    ]
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
    response.headers["x-strack-llm-pipeline-version"] = STRACK_LLM_PIPELINE_VERSION
    response.headers["min-strack-version"] = STRACK_MIN_VERSION
    return response

create_web_dir()

app.mount("/public", StaticFiles(directory="public"), name="public")
app.mount("/", StaticFiles(directory="app"), name="app")

def main():
    scan_models()
    logger = logging.getLogger(__name__)
    logger.info("服务启动 port=8000 version=%s", STRACK_LLM_PIPELINE_VERSION)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()