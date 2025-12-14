import os
import json
import requests  # <--- 核心改变：用最原始的 requests 库

# --- 1. 强制离线模式 ---
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- 2. 路径设置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, "../data/chroma_db")


class RAGSystem:
    def __init__(self):
        print("正在初始化 RAG 系统...")

        # A. 向量模型
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        # B. 向量数据库
        if not os.path.exists(DB_DIR):
            raise FileNotFoundError(f"找不到数据库目录: {DB_DIR}")

        self.vector_db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=self.embedding_model
        )
        print("✅ 系统初始化完成！")

    def query(self, question):
        print(f"\n🔍 正在检索：{question}")

        # --- 步骤 1: 检索 (这里先恢复为 k=3，如果卡顿再改回 k=1) ---
        docs = self.vector_db.similarity_search(question, k=3)
        if not docs:
            print("⚠️ 未找到相关文档。")
            return

        print("\n📚 检索到的参考资料：")
        context_text = ""
        for i, doc in enumerate(docs):
            content = doc.page_content.replace("\n", " ")
            print(f"[{i + 1}] {content[:50]}...")
            # 限制长度防止爆显存
            context_text += f"片段{i + 1}: {content[:500]}\n"

        # --- 步骤 2: 构建 Prompt ---
        system_prompt = "你是一个专业助手。请根据【参考资料】回答问题。如果不知道就只根据【问题】来回答。"
        user_prompt = f"【参考资料】:\n{context_text}\n\n【问题】:\n{question}"

        # --- 步骤 3: 调用 LLM (使用 requests 暴力直连) ---
        print("\n🤖 DeepSeek 正在思考...")

        url = "http://127.0.0.1:1234/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "stream": True  # 开启流式
        }

        try:
            # proxies={"http": None, "https": None} 是核武器
            # 它强制 requests 库完全忽略系统的任何代理设置
            response = requests.post(
                url,
                headers=headers,
                json=data,
                stream=True,
                proxies={"http": None, "https": None},
                timeout=60
            )

            # 检查状态码
            if response.status_code != 200:
                print(f"❌ 服务器返回错误: {response.status_code}")
                print(response.text)
                return None

            return response

        except Exception as e:
            print(f"\n❌ 连接失败: {e}")
            return None


if __name__ == "__main__":
    rag = RAGSystem()

    test_question = input("请输入测试问题:")

    # 获取 response 对象
    response = rag.query(test_question)

    if response:
        print("\n📢 回答：")
        import time

        # 手动解析流式数据 (Parsing SSE)
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                # 过滤掉 "data: " 前缀
                if decoded_line.startswith("data: "):
                    json_str = decoded_line[6:]  # 去掉前6个字符
                    if json_str.strip() == "[DONE]":
                        break
                    try:
                        json_data = json.loads(json_str)
                        content = json_data['choices'][0]['delta'].get('content', '')
                        if content:
                            for char in content:
                                print(char, end="", flush=True)
                                time.sleep(0.01)
                    except json.JSONDecodeError:
                        continue
        print("\n")