import chainlit as cl
import os
import sys
import json
import textwrap  # <--- [新增] 用于处理多行字符串缩进

# --- 1. 路径设置 ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_core02 import RAGSystem
from ingest import create_vector_db, reset_vector_db


# --- 2. 工具函数 ---

def get_file_list_info():
    """读取 docs 文件夹，返回格式化的文件列表字符串"""
    save_dir = os.path.join(os.path.dirname(__file__), "data/docs")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    files = os.listdir(save_dir)
    if not files:
        return "📂 **当前知识库为空**"

    info = f"📂 **当前知识库 ({len(files)} 个文件):**\n"
    for f in files:
        info += f"- `{f}`\n"
    return info


async def send_control_panel(title="控制面板"):
    """
    [新增] 封装发送控制面板的逻辑
    在启动时、操作完成后调用，确保按钮永远可见
    """
    file_info = get_file_list_info()

    actions = [
        cl.Action(name="rebuild_db", payload={"value": "rebuild"}, label="🔄 上传并重建",
                  description="上传新文件并更新数据库"),
        cl.Action(name="reset_db", payload={"value": "reset"}, label="🗑️ 清空所有", color="red",
                  description="删除文件和数据库")
    ]

    # [修改] 使用 textwrap.dedent 去除缩进，解决 Markdown 不渲染的问题
    content = textwrap.dedent(f"""
        # {title}

        {file_info}

        ---
        👇 **请选择操作：**
    """).strip()

    await cl.Message(content=content, actions=actions).send()


# --- 3. 初始化 RAG 系统 ---
rag_system = None


def get_rag():
    global rag_system
    if rag_system is None:
        print("正在启动 RAG 引擎...")
        rag_system = RAGSystem()
    return rag_system


# --- 4. 启动逻辑 & 右侧设置栏 ---
@cl.on_chat_start
async def start():
    # 配置右侧设置栏
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="Mode",
                label="🧠 检索模式 (Search Mode)",
                values=["Flash (极速)", "Pro (深度)"],
                initial_index=0,
                description="Flash: 直接检索 Top-3; Pro: 检索 Top-10 并使用 Rerank 重排序。"
            ),
        ]
    ).send()
    cl.user_session.set("settings", settings)

    # [修改] 调用封装好的函数发送面板
    await send_control_panel(title="👋 欢迎使用本地 RAG 助手")


# --- 5. 监听设置变更 ---
@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("settings", settings)
    current_mode = settings["Mode"]
    await cl.Message(content=f"⚙️ 模式已切换为: **{current_mode}**", author="System").send()


# --- 6. 按钮动作处理 ---
@cl.action_callback("rebuild_db")
async def on_rebuild(action):
    # 1. 请求上传
    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="请上传文档 (PDF/TXT/MD/DOCX/CSV)，上传完成后自动开始构建。",
            accept=["application/pdf", "text/plain", "text/markdown",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/csv"],
            max_size_mb=20,
            max_files=10,
            timeout=600
        ).send()

    # 2. 保存文件
    save_dir = os.path.join(os.path.dirname(__file__), "data/docs")

    # 发送一个临时状态消息
    status_msg = cl.Message(content=f"正在保存 {len(files)} 个文件...")
    await status_msg.send()

    for file in files:
        save_path = os.path.join(save_dir, file.name)
        if hasattr(file, "path"):
            import shutil
            shutil.copy(file.path, save_path)
        elif hasattr(file, "content"):
            with open(save_path, "wb") as f:
                if isinstance(file.content, str):
                    f.write(file.content.encode('utf-8'))
                else:
                    f.write(file.content)

    # 3. 调用后端
    status_msg.content = "⏳ 正在构建向量索引 (ingesting)..."
    await status_msg.update()

    success, result_msg = await cl.make_async(create_vector_db)()

    # 4. 反馈结果
    if success:
        global rag_system
        rag_system = RAGSystem()  # 强制重载
        status_msg.content = f"✅ **构建成功！**\n> {result_msg}"
    else:
        status_msg.content = f"❌ **构建失败**: {result_msg}"

    await status_msg.update()

    # [关键修改] 操作完成后，再次发送控制面板，方便用户继续操作
    await send_control_panel(title="📂 知识库已更新")


@cl.action_callback("reset_db")
async def on_reset(action):
    status_msg = cl.Message(content="⏳ 正在清空文件和数据库...")
    await status_msg.send()

    save_dir = os.path.join(os.path.dirname(__file__), "data/docs")
    if os.path.exists(save_dir):
        for f in os.listdir(save_dir):
            try:
                os.remove(os.path.join(save_dir, f))
            except:
                pass

    success, result_msg = await cl.make_async(reset_vector_db)()

    if success:
        status_msg.content = "✅ **已恢复出厂设置**。所有数据已清除。"
    else:
        status_msg.content = f"❌ 清空失败: {result_msg}"
    await status_msg.update()

    # [关键修改] 操作完成后，再次发送控制面板
    await send_control_panel(title="🗑️ 知识库已清空")


# --- 7. 核心对话逻辑 ---
@cl.on_message
async def main(message: cl.Message):
    rag = get_rag()

    settings = cl.user_session.get("settings")
    mode_str = settings["Mode"] if settings else "Flash (极速)"
    mode_param = "pro" if "Pro" in mode_str else "flash"

    history = []
    msg = cl.Message(content="")

    thinking_step = cl.Step(name="DeepSeek 深度思考", type="run")
    thinking_step.language = "text"

    response_obj, source_docs, intent = await cl.make_async(rag.query)(
        message.content,
        history=history,
        mode=mode_param
    )

    if intent == "SEARCH":
        await thinking_step.send()

    if response_obj:
        is_thinking = False
        has_started_thinking = False

        for line in response_obj.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    json_str = decoded_line[6:]
                    if json_str.strip() == "[DONE]": break
                    try:
                        json_data = json.loads(json_str)
                        content = json_data['choices'][0]['delta'].get('content', '')

                        if content:
                            if "<think>" in content:
                                is_thinking = True
                                has_started_thinking = True
                                content = content.replace("<think>", "")
                                if not content: continue

                            if "</think>" in content:
                                is_thinking = False
                                part_thought = content.split("</think>")[0]
                                thinking_step.output += part_thought
                                await thinking_step.update()

                                part_answer = content.split("</think>")[1]
                                await msg.stream_token(part_answer)
                                continue

                            if is_thinking:
                                thinking_step.output += content
                                await thinking_step.stream_token(content)
                            else:
                                await msg.stream_token(content)

                    except Exception:
                        continue

        if has_started_thinking:
            await thinking_step.update()

        # 来源折叠显示
        if source_docs:
            ref_text = "\n\n<details><summary><b>📚 参考来源 (点击展开)</b></summary>\n\n"
            for i, doc in enumerate(source_docs):
                source_name = os.path.basename(doc.metadata.get("source", "未知文件"))
                page = doc.metadata.get("page", 0) + 1
                text_preview = doc.page_content[:200].replace("\n", " ") + "..."
                ref_text += f"**[{i + 1}] {source_name}** (P{page})\n> {text_preview}\n\n"
            ref_text += "</details>"
            msg.content += ref_text
            await msg.update()

        await msg.send()
    else:
        await cl.Message(content="❌ 连接超时或未找到答案。").send()