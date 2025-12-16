import streamlit as st
import os
import sys
import time
import json
# --- 关键：将 src 目录加入 Python 搜索路径 ---
# 这样 app.py 才能找到 src 下的 modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_core02 import RAGSystem
from ingest import create_vector_db, reset_vector_db

# --- 页面配置 ---
st.set_page_config(
    page_title="RAG 知识库助手",
    page_icon="",
    layout="wide"
)

# --- 标题与简介 ---
st.title("本地化 RAG 个人知识库")
st.markdown("基于 **DeepSeek-14B** + **ChromaDB** 构建的私有知识助手")

# --- 初始化 RAG 系统 (利用 Streamlit 的缓存机制) ---
# @st.cache_resource 确保 RAGSystem 只被初始化一次，
# 不会因为用户每次点按钮都重新加载模型（那样会很慢）
@st.cache_resource
def load_rag_system():
    return RAGSystem()

try:
    with st.spinner("正在启动引擎，加载知识库..."):
        rag = load_rag_system()
    st.success("✅ 系统就绪！")

except Exception as e:
    st.warning("⚠️ 尚未检测到知识库，请在左侧上传文件并点击'重建'。")

# --- 侧边栏：功能区 ---
with st.sidebar:
    st.header("⚙️ 知识库管理")
    st.info(
        "当前模型: DeepSeek-R1-Distill-Qwen-14B\n"
        "运行模式: Local (LM Studio)\n"
        "检索策略: Top-3 混合检索"
    )

    # 检索模式选择器
    st.write("---")
    st.write("🧠 **检索模式**")
    search_model = st.radio(
        "选择思考深度:",
        ("Flash (极速)", "Pro (深度)"),
        index=0,
        help="Flash: 仅使用向量检索，速度快。\nPro: 引入 BGE 重排序模型，精准度高但稍慢。"
    )
    # 将中文选项映射回代码用的参数值
    mode_param = "flash" if "Flash" in search_model else "pro"

    st.divider()

    # 1. 定义保存路径
    save_dir = os.path.join(os.path.dirname(__file__), 'data/docs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- 功能区: 显示当前文件列表 ---
    current_files = os.listdir(save_dir)
    st.info(f"📚 待处理文档: {len(current_files)} 个")

    with st.expander("查看当前文件列表"):
        if len(current_files) == 0:
            st.text("(空)")
        else:
            for f in current_files:
                st.text(f"- {f}")

    # --- 功能区: 恢复出厂设置按钮 ---
    # 只有当有文件时才显示，或者常驻显示
    if st.button("🗑️ 恢复出厂设置 (清空文件+数据库)", type="primary"):
        try:
            # 1. 释放资源 (断开数据库连接)
            st.cache_resource.clear()

            # 2. 清空 data/docs 文件夹
            deleted_count = 0
            for filename in current_files:
                file_path = os.path.join(save_dir, filename)
                os.remove(file_path)
                deleted_count += 1
            st.write(f"✅ 已删除 {deleted_count} 个本地文件。")

            # 3. 清空向量数据库 (调用新函数)
            success, msg = reset_vector_db()
            if success:
                st.success("所有数据已清空！页面即将刷新...")
                time.sleep(1.5)
                st.rerun()
            else:
                st.error(msg)

        except Exception as e:
            st.error(f"操作失败: {e}")

    st.divider()

    # --- 文件上传区 ---
    uploaded_file = st.file_uploader(
        "上传新文档",
        help="追加模式,文件大于20MB,CPU死给你看",
        type=["pdf", "txt", "docx", "md", "csv"],
        accept_multiple_files=True,

    )

    if uploaded_file:
        new_count = 0
        for uploaded_file in uploaded_file:
            save_path = os.path.join(save_dir, uploaded_file.name)
            # 判断：只有文件不存在时才写入，避免重复IO
            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                new_count += 1

        if new_count > 0:
            st.success(f"已上传 {new_count} 个新文件。")
            time.sleep(0.5)
            st.rerun()  # 刷新以更新上方的列表
    st.divider()

    # --- 触发重建按钮 ---
    if st.button("🔄 重建知识库 (Process)"):
        with st.status("正在处理数据...", expanded=True) as status:
            st.write("1. 正在初始化...")

            # 清除缓存
            st.cache_resource.clear()

            st.write("2. 正在更新数据库 (逻辑清空模式)...")
            success, msg = create_vector_db()

            if success:
                st.write("3. 数据加载完成!")
                status.update(label="✅ 知识库构建成功！", state="complete", expanded=False)
                st.success(msg)
                time.sleep(1)
                st.rerun()
            else:
                status.update(label="❌ 构建失败", state="error")
                st.error(msg)



# --- 主聊天界面 ---

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("请输入你的问题（关于已上传的文档）..."):
    # 1. 显示用户问题
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2.调用 RAG 获取回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # 增加安全性检查：
        # 如果刚才点击了重建，rag 对象可能被清理了，这里尝试重新加载
        try:
            if 'rag' not in locals():
                rag = load_rag_system()

                # 显示“正在思考”状态
            with st.spinner(f"DeepSeek ({mode_param} mode) 正在阅读文档并思考..."):
                # 调用在 rag_core.py 里写的 query 方法并将 mode_param 传进去
                response_obj = rag.query(prompt, mode=mode_param)


            if response_obj:
                # --- 流式渲染核心逻辑: 支持思考过程折叠 ---
                # 对应 requests 的手动解析逻辑，Streamlit 会实时刷新界面

                # 1. 预先创建 UI 占位符
                # 创建一个默认折叠的扩展框，用于放思考过程
                status_container = st.status("💭 DeepSeek 正在深度思考...", expanded=False, state="running")

                thought_placeholder = status_container.empty()

                # 创建主回答的占位符
                answer_placeholder = st.empty()

                # 2. 初始化变量
                full_content = ""      # 原始完整内容（包含标签）
                thought_content = ""   # 思考部分的内容
                answer_content = ""    # 正文回答的内容
                is_thinking = False    # 标记当前是否处于思考模式

                import json
                for line in response_obj.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")
                        if decoded_line.startswith("data: "):
                            json_str = decoded_line[6:]
                            if json_str.strip() == "[DONE]":
                                break
                            try:
                                json_data = json.loads(json_str)
                                content = json_data['choices'][0]['delta'].get('content', '')

                                if content:
                                    full_content += content

                                    # --- 智能分流逻辑 ---

                                    # 检查是否包含思考结束标签
                                    if "</think>" in full_content:
                                        is_thinking = False

                                        # 将完整内容按标签切分
                                        parts = full_content.split("</think>")

                                        # 第一部分是思考内容 (去掉开头的 <think>)
                                        thought_part = parts[0].replace("<think>", "").strip()

                                        # 第二部分: 正式回答
                                        answer_part = parts[1]

                                        # 更新 UI
                                        thought_placeholder.markdown(thought_part)
                                        answer_placeholder.markdown(answer_part + "|")

                                        status_container.update(label= "深度思考过程", state="complete", expanded=False)

                                        # 同步最终变量，方便存入历史记录
                                        thought_content = thought_part
                                        answer_content = answer_part

                                    elif "<think>" in full_content:
                                        # 如果有开始标签，但还没结束，说明正在思考中
                                        is_thinking = True

                                        # 提取思考内容 (去掉开头的 <think>)
                                        current_thought = full_content.replace("<think>", "")
                                        thought_placeholder.markdown(current_thought + "|")

                                    else:
                                        # 如果完全没有标签 (有的模型可能不输出 think 标签)，直接当做正文

                                        answer_content += content
                                        answer_placeholder.markdown(answer_content + "|")

                                        # status_container.update(label="(本次回答无思考过程)", state="complete", expanded=False)
                            except Exception:
                                continue

                # 4. 最终收尾
                # 把光标去掉，显示最终结果
                if thought_content:
                    thought_placeholder.markdown(thought_content)

                status_container.update(label="深度思考过程", state="complete", expanded=False)
                answer_placeholder.markdown(answer_content)

                # 5. [关键] 保存到历史记录
                # 我们只保存“正文回答”到历史记录，防止历史记录太长，且思考过程通常不用于下一轮对话
                # 如果你想连思考过程一起存，可以存 full_content
                final_save_content = answer_content if answer_content else full_content
                st.session_state.messages.append({"role": "assistant", "content": final_save_content})

            else:
                st.error("连接超时或未找到答案，请检查 LM Studio。")
        except Exception as e:
            st.error(f"发生错误: {e}")
            st.info("💡 提示：如果是刚重建完知识库，请尝试刷新页面。")



