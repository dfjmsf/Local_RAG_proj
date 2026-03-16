"""
FastAPI 后端服务 (server.py)
功能: 会话管理 / 流式聊天 / 文件管理 / 知识库重建
"""

import os
import sys
import json
import shutil
import asyncio
import threading
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_core02 import RAGSystem
from ingest import create_vector_db, reset_vector_db
import database as db

# ============================================================
# 配置
# ============================================================

os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "False"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(CURRENT_DIR, "data/docs")
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

# ============================================================
# 初始化
# ============================================================

app = FastAPI(title="Local RAG API", version="1.0")
db.init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system = RAGSystem()


class ChatRequest(BaseModel):
    question: str
    history: List[dict] = []
    mode: str = "flash"
    session_id: Optional[str] = None


# ============================================================
# 1. 会话管理接口
# ============================================================

@app.get("/api/sessions")
async def get_sessions():
    """获取会话列表"""
    return db.get_all_sessions()


@app.post("/api/sessions")
async def create_new_session():
    """创建一个新会话"""
    session_id = db.create_session(title="新对话")
    return {"id": session_id, "title": "新对话", "created_at": datetime.now().isoformat()}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    db.delete_session(session_id)
    return {"status": "success"}


@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    """获取指定会话的消息记录"""
    return db.get_session_messages(session_id)


@app.put("/api/sessions/{session_id}")
async def update_session(session_id: str, payload: dict):
    """更新标题 (Payload: {"title": "..."})"""
    new_title = payload.get("title")
    if new_title:
        db.update_session_title(session_id, new_title)
    return {"status": "success"}


# ============================================================
# 2. 核心对话接口
# ============================================================

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # 保存用户消息到数据库
    if request.session_id:
        db.add_message(
            session_id=request.session_id,
            role="user",
            content=request.question
        )
        # 新对话第一句自动重命名标题
        if not request.history:
            db.update_session_title(request.session_id, request.question[:20])

    async def event_generator():
        full_content = ""
        full_thought = ""

        try:
            response, docs, intent = await run_in_threadpool(
                rag_system.query,
                question=request.question,
                history=request.history,
                mode=request.mode
            )

            # 发送意图
            yield json.dumps({"type": "intent", "data": intent}, ensure_ascii=False) + "\n"
            await asyncio.sleep(0)

            # 发送参考资料
            serialized_docs = []
            if docs:
                for d in docs:
                    serialized_docs.append({
                        "source": os.path.basename(d.metadata.get("source", "unknown")),
                        "page": d.metadata.get("page", 0) + 1,
                        "content": d.page_content
                    })
                yield json.dumps({"type": "sources", "data": serialized_docs}, ensure_ascii=False) + "\n"
                await asyncio.sleep(0)

            # 流式发送 LLM 输出
            if response:
                raw_buffer = ""
                queue = asyncio.Queue()
                loop = asyncio.get_event_loop()

                def _stream_reader():
                    """后台线程读取同步流，通过 queue 桥接到异步 generator"""
                    try:
                        for line in response.iter_lines():
                            if line:
                                loop.call_soon_threadsafe(queue.put_nowait, line)
                    except Exception as e:
                        loop.call_soon_threadsafe(queue.put_nowait, e)
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                reader_thread = threading.Thread(target=_stream_reader, daemon=True)
                reader_thread.start()

                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        print(f"Stream Error: {item}")
                        break

                    decoded_line = item.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[6:]
                        if json_str.strip() == "[DONE]":
                            break
                        try:
                            json_data = json.loads(json_str)
                            content = json_data['choices'][0]['delta'].get('content', '')
                            if content:
                                raw_buffer += content

                                # 通过全局 buffer 判断 <think> 标签状态
                                if "</think>" in raw_buffer:
                                    parts = raw_buffer.split("</think>", 1)
                                    full_thought = parts[0].replace("<think>", "")
                                    full_content = parts[1]
                                elif "<think>" in raw_buffer:
                                    full_thought = raw_buffer.replace("<think>", "")
                                    full_content = ""
                                else:
                                    full_thought = ""
                                    full_content = raw_buffer

                                yield json.dumps({"type": "content", "data": content}, ensure_ascii=False) + "\n"
                                await asyncio.sleep(0)
                        except Exception:
                            continue

                # 流式结束后保存 AI 回答
                if request.session_id:
                    clean_thought = full_thought.replace("<think>", "").replace("</think>", "")
                    clean_content = full_content.replace("</think>", "").strip()
                    db.add_message(
                        session_id=request.session_id,
                        role="assistant",
                        content=clean_content,
                        thought=clean_thought if clean_thought else None,
                        sources=serialized_docs if serialized_docs else None
                    )
            else:
                yield json.dumps({"type": "error", "data": "LLM 未返回响应"}, ensure_ascii=False) + "\n"

        except Exception as e:
            print(f"Server Error: {e}")
            yield json.dumps({"type": "error", "data": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


# ============================================================
# 3. 文件管理接口
# ============================================================

@app.get("/api/files")
async def list_files():
    if not os.path.exists(DOCS_DIR):
        physical_files = set()
    else:
        physical_files = set(os.listdir(DOCS_DIR))
    indexed_files_in_db = await run_in_threadpool(rag_system.get_indexed_files)
    response_data = {"indexed": [], "pending": []}
    for f in physical_files:
        if f in indexed_files_in_db:
            response_data["indexed"].append(f)
        else:
            response_data["pending"].append(f)
    response_data["indexed"].sort()
    response_data["pending"].sort()
    return response_data


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    saved_files = []
    for file in files:
        file_path = os.path.join(DOCS_DIR, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
        except Exception as e:
            return JSONResponse(status_code=500, content={"message": f"上传失败 {file.filename}: {str(e)}"})
    return {"message": f"成功上传 {len(saved_files)} 个文件", "files": saved_files}


@app.post("/api/rebuild")
async def rebuild_db():
    success, msg = await run_in_threadpool(create_vector_db)
    if success:
        global rag_system
        rag_system = RAGSystem()
        return {"status": "success", "message": msg}
    else:
        return JSONResponse(status_code=500, content={"status": "error", "message": msg})


@app.post("/api/reset")
async def reset_db():
    try:
        if os.path.exists(DOCS_DIR):
            for filename in os.listdir(DOCS_DIR):
                file_path = os.path.join(DOCS_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"删除失败: {e}")
        success, msg = await run_in_threadpool(reset_vector_db)
        global rag_system
        rag_system = RAGSystem()
        if success:
            return {"status": "success", "message": "已清空文件和数据库"}
        else:
            return JSONResponse(status_code=500, content={"status": "error", "message": msg})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 启动 FastAPI 后端服务...")
    uvicorn.run(app, host="127.0.0.1", port=8000)