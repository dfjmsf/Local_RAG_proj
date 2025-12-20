import os
import json
import requests  # <--- 核心改变：用最原始的 requests 库

# --- 1. 强制离线模式 ---
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

# --- 2. 路径设置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, "../data/chroma_db")
RERANK_MODEL_PATH = os.path.join(CURRENT_DIR, "../model_cache/bge-reranker-base")


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

        # C. 初始化 Rerank 模型
        print(f" -> 正在加载 Rerank 模型 ({RERANK_MODEL_PATH})...")
        try:
            # device="cpu" 保证兼容性，有 N 卡可以改成 "cuda"
            self.reranker = CrossEncoder(RERANK_MODEL_PATH, device="cpu")
            print(" -> Rerank 模型加载成功！")
        except Exception as e:
            print(f"❌ Rerank 模型加载失败: {e}")
            print("   (将自动降级为仅使用向量检索)")
            self.reranker = None

        print("✅ 系统初始化完成！")

    def query(self, question, history=[], mode="flash"):
        """
        :param question: 用户问题
        :param mode: 'flash' (极速) 或 'pro' (深度)
        :param history: 前端传来的历史对话列表 (list of dict)
        :return: (response对象, 参考文档列表)
        """
        print(f"\n🔍 正在检索：{question} | 模式: {mode.upper()}")

        final_docs = []

        # --- 步骤 1: 检索策略分流 ---
        if mode == "pro" and self.reranker:
            # === Pro 模式 (深度) ===
            # 1. 扩大召回：先捞出 20 条 (Top-20)
            initial_docs = self.vector_db.similarity_search(question, k=20)

            if initial_docs:
                # 2. 准备配对数据 [问题, 文档内容]
                pairs = [[question, doc.page_content] for doc in initial_docs]

                # 3.模型打分
                print(" -> 正在进行 Rerank 重排序...")
                scores = self.reranker.predict(pairs)

                # 4. 排序截断 (Top-3)
                # 将文档和分数打包，按分数降序排
                scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)

                print("\n📊 Rerank 打分结果 (Top-5):")
                for doc, score in scored_docs[:5]:
                    print(f"   [分: {score:.4f}] {doc.page_content[:30]}...")

                # 取前 3 名的文档对象
                final_docs = [doc for doc, score in scored_docs[:5]]
            else:
                print("⚠️ 初步检索未找到文档。")

        else:
            # === Flash 模式 (极速) ===
            # 直接找 Top-3，不经过模型重算，速度最快
            final_docs = self.vector_db.similarity_search(question, k= 5)


        # --- 通用逻辑 ---
        if not final_docs:
            print("⚠️ 未找到相关文档。")
            return None, []

        print("\n📚 最终参考资料：")
        context_text = ""
        for i, doc in enumerate(final_docs):
            content = doc.page_content.replace("\n", " ")
            print(f"[{i + 1}] {content[:50]}...")
            # 限制长度防止爆显存
            context_text += f"片段{i + 1}: {content[:500]}\n"

        # --- 步骤 2: 构建 Prompt 与 历史消息注入 ---
        # 1. 定义系统提示词 (Persona)
        system_prompt = "你是一个专业助手。请根据【参考资料】回答问题。如果不知道就说不知道。在回答之前请针对用户的问题与要求对用户进行简短的夸奖"

        # 2. 初始化消息列表
        messages_payload = [
            {"role": "system", "content": system_prompt}
        ]

        # 3. 注入历史记忆 (Sliding Window)
        # 只保留最近的 6 条消息 (即 3 轮对话)，防止上下文超限
        # history 格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        if history:
            recent_history = history[-6:]
            for msg in recent_history:
                messages_payload.append(msg)
            print(f" -> 已注入历史记忆: {len(recent_history)} 条消息")

        # 4. 拼接当前最新的 User Prompt (包含 RAG 上下文)
        current_user_prompt = f"【参考资料】:\n{context_text}\n\n【问题】:\n{question}"
        messages_payload.append({"role": "user", "content": current_user_prompt})

        # --- 步骤 3: 调用 LLM (使用 requests 暴力直连) ---
        print("\n🤖 DeepSeek 正在思考...")

        url = "http://127.0.0.1:1234/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "local-model",
            "messages": messages_payload,
            "temperature": 0.7,
            "stream": True  # 开启流式输出
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
                return None, []

            return response, final_docs

        except Exception as e:
            print(f"\n❌ 连接失败: {e}")
            return None, []


if __name__ == "__main__":
    rag = RAGSystem()

    # 获取 response 对象
    test_question = input("请输入测试问题:")
    response = rag.query(test_question, mode="pro")  # 默认测试 Pro 模式



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