import streamlit as st
import os
import sys

# --- 关键：将 src 目录加入 Python 搜索路径 ---
# 这样 app.py 才能找到 src 下的 modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_core02 import RAGSystem

# --- 页面配置 ---
st.set_page_config(
    page_title="RAG 知识库助手",
    page_icon="🤖",
    layout="wide"
)

# --- 标题与简介 ---
st.title("🤖 本地化 RAG 个人知识库")
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
    st.error(f"系统启动失败: {e}")
    st.stop()

# --- 侧边栏：功能区 ---
with st.sidebar:
    st.header("⚙️ 设置")
    st.info(
        "当前模型: DeepSeek-R1-Distill-Qwen-14B\n"
        "运行模式: Local (LM Studio)\n"
        "检索策略: Top-3 混合检索"
    )

    # 这里未来可以加“上传文件”功能，也就是调用 ingest.py
    if st.button("重新构建知识库(Ingest)"):
        st.warning("目前请在后台运行 ingest.py 手动更新数据。")

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

    # 显示“正在思考”状态
    with st.spinner("DeepSeek 正在阅读文档并思考..."):
        # 调用在 rag_core.py 里写的 query 方法
        response_obj = rag.query(prompt)

    if response_obj:
        # --- 流式渲染核心逻辑 ---
        # 对应 requests 的手动解析逻辑，Streamlit 会实时刷新界面
        import json
        for line in response_obj.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data:"):
                    json_str = decoded_line[6:]
                    if json_str.strip() == "[DONE]":
                        break
                    try:
                        json_data = json.loads(json_str)
                        content = json_data['choices'][0]['delta'].get('content', '')
                        if content:
                            full_response += content
                            # 实时更新 UI
                            message_placeholder.markdown(full_response + "|")
                    except Exception as e:
                        print(e)

        message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    else:
        st.error("连接超时或未找到答案，请检查 LM Studio。")



