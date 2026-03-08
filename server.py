import os
import shutil
import json
import asyncio
import threading
from typing import List, Optional  # <--- 引入 Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_core02 import RAGSystem
from ingest import create_vector_db, reset_vector_db
# [新增] 引入数据库模块
import database as db

# --- 禁用遥测 ---
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "False"

# --- 配置路径 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(CURRENT_DIR, "data/docs")
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

# --- 初始化 ---
app = FastAPI(title="Local RAG API", version="1.0")
# 确保数据库已初始化
db.init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system = RAGSystem()


# [修改] 请求模型增加 session_id
class ChatRequest(BaseModel):
    question: str
    history: List[dict] = []
    mode: str = "flash"
    session_id: Optional[str] = None  # 可选，如果有值则保存记录


# ===========================
# 1. 会话管理接口 (Session API)
# ===========================

@app.get("/api/sessions")
async def get_sessions():
    """获取会话列表"""
    return db.get_all_sessions()


@app.post("/api/sessions")
async def create_new_session():
    """创建一个新会话"""
    session_id = db.create_session(title="新对话")
    from datetime import datetime
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


# ===========================
# 2. 核心对话接口 (集成保存逻辑)
# ===========================
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # [新增] 1. 如果有 session_id，先把用户的问题存入数据库
    if request.session_id:
        db.add_message(
            session_id=request.session_id,
            role="user",
            content=request.question
        )

        # [可选优化] 如果是新对话的第一句，自动用问题重命名标题
        # 简单判断：如果历史为空，且是第一句
        if not request.history:
            # 截取前20个字作为标题
            new_title = request.question[:20]
            db.update_session_title(request.session_id, new_title)

    async def event_generator():
        # 用于收集 AI 的完整回答，以便最后存库
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

            # 发送资料
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

            # 发送内容
            if response:
                # [修复] 使用 raw_buffer 累积全文，避免 <think> 标签被跨 chunk 切割
                raw_buffer = ""  # 累积 LLM 的全部原始输出

                # [修复] 使用 asyncio.Queue + 后台线程桥接同步 iter_lines()
                # 避免同步阻塞事件循环
                queue = asyncio.Queue()
                loop = asyncio.get_event_loop()

                def _stream_reader():
                    """在后台线程中读取同步流，通过 queue 传递给异步 generator"""
                    try:
                        for line in response.iter_lines():
                            if line:
                                loop.call_soon_threadsafe(queue.put_nowait, line)
                    except Exception as e:
                        loop.call_soon_threadsafe(queue.put_nowait, e)
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)  # 哨兵值，表示结束

                reader_thread = threading.Thread(target=_stream_reader, daemon=True)
                reader_thread.start()

                while True:
                    item = await queue.get()
                    if item is None:
                        break  # 流读取完毕
                    if isinstance(item, Exception):
                        print(f"Stream Error: {item}")
                        break

                    decoded_line = item.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[6:]
                        if json_str.strip() == "[DONE]": break
                        try:
                            json_data = json.loads(json_str)
                            content = json_data['choices'][0]['delta'].get('content', '')
                            if content:
                                # 累积原始输出到 buffer
                                raw_buffer += content

                                # [修复] 通过全局 buffer 判断 think 标签状态
                                # 提取结束标签后的正文内容，和标签内的思考内容
                                if "</think>" in raw_buffer:
                                    # 标签已闭合：提取思考和正文
                                    parts = raw_buffer.split("</think>", 1)
                                    full_thought = parts[0].replace("<think>", "")
                                    full_content = parts[1]
                                elif "<think>" in raw_buffer:
                                    # 标签未闭合：全部是思考内容
                                    full_thought = raw_buffer.replace("<think>", "")
                                    full_content = ""
                                else:
                                    # 无标签：全部是正文
                                    full_thought = ""
                                    full_content = raw_buffer

                                yield json.dumps({"type": "content", "data": content}, ensure_ascii=False) + "\n"
                                await asyncio.sleep(0)
                        except Exception:
                            continue

                # [新增] 2. 流式结束后，把 AI 的回答存入数据库
                if request.session_id:
                    # 清洗一下 thought 里的标签
                    clean_thought = full_thought.replace("<think>", "").replace("</think>", "")
                    # 清洗一下 content 里的标签 (有时候 </think> 会漏在 content 里)
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


# ... (文件管理接口保持不变) ...
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


if __name__ == "__main__":
    import uvicorn

    print("🚀 启动 FastAPI 后端服务...")
    uvicorn.run(app, host="127.0.0.1", port=8000)