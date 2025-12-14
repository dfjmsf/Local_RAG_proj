import os
import sys
import httpx

# 这行代码告诉库：直接读本地缓存，绝对不要尝试连接 huggingface.co
os.environ["HF_HUB_OFFLINE"] = "1"

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, '../data/chroma_db')

class RAGSystem:
    def __init__(self):
        """
                初始化 RAG 系统：加载向量数据库和 LLM 客户端
        """
        print("正在初始化 RAG 系统...")

        # A. 准备向量模型 (必须和 ingest.py 用的一模一样！)
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        # B. 加载已经存在的向量数据库
        # 知识点：持久化加载
        # 只要指定 persist_directory，Chroma 就会自动读取硬盘上的数据，不需要重新 ingest
        if not os.path.exists(DB_DIR):
            raise FileNotFoundError(f"找不到数据库目录: {DB_DIR}。请先运行 ingest.py！")

        self.vector_db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=self.embeddings_model
        )

        # C. 连接本地 LLM
        print(" -> 正在配置网络连接 (强制绕过代理)...")

        # 使用 trust_env=False
        # 这意味着：不读取系统环境变量（即忽略 VPN/Proxy 设置），直接直连。
        # 这是兼容性最好的“绕过代理”的方法
        http_client = httpx.Client(
            trust_env=False,
            timeout=120.0  # 设置极长的超时时间 (120秒)
        )
        self.client = OpenAI(base_url="https://127.0.0.1:1234/v1",
                             api_key="lm-studio",
                             http_client=http_client
                             )
        print("✅ 系统初始化完成！")

    def query(self, question):
        """
               核心方法：执行 RAG 流程
        """
        print(f"\n🔍 正在检索：{question}")

        # --- 步骤 1: 语义检索 (Retrieval) ---
        # k=3 表示我们要找最相似的 3 个片段
        # 知识点：Similarity Search (相似度搜索)
        # 它会将 question 转为向量，计算与数据库中所有向量的“距离”，返回最近的 k 个。
        docs = self.vector_db.similarity_search(question, k=3)

        if not docs:
            print("⚠️ 未找到相关文档。")
            return

        # 打印一下找到了什么（调试用，让你看到 RAG 到底搜到了啥,真正使用的时候最好关(注释)掉）
        print("\n📚 检索到的参考资料：")
        context_text = ""
        for i, doc in enumerate(docs):
            # source 是文件路径，page 是页码
            sourcs = os.path.basename(doc.metadata.get("source", "未知来源"))
            page = doc.metadata.get("page", 0) + 1
            content = doc.page_content.replace("\n", " ")   # 去掉换行，让显示更紧凑

            print(f"[{i+1}]...{content[:50]}... (来源: {sourcs} 第{page}页)")

            context_text += f"片段{i+1}: {content}\n"

        # --- 步骤 2: 构建 Prompt (Prompt Engineering) ---
        # 这是一个经典的 RAG 提示词模版
        system_prompt = """你是一个专业的知识库助手。
请完全根据下面的【参考资料】来回答用户的问题。
如果【参考资料】里没有提到答案，就直接说“我在现有文档中找不到答案”，不要编造。
请用中文回答。
"""
        user_prompt = f"""
【参考资料】：
{context_text}

【用户问题】：
{question}
        """

        # --- 步骤 3: 调用 LLM 生成回答 ---
        print("\n DeepSeek 正在思考...")
        try:
            response = self.client.chat.completions.create(
                model="local-modal",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, # RAG 任务要低创造性，保证严谨
                stream=True
            )
            # 返回一个生成器对象，让 UI 层去处理流式输出
            return response
        except Exception as e:
            # --- 详细的错误打印 ---
            print(f"\n❌ 调用失败！详细错误信息:")
            print(f"类型: {type(e).__name__}")
            print(f"内容: {str(e)}")
            return None

if __name__ == "__main__":
    # 实例化系统
    rag = RAGSystem()

    # 这里的测试问题，请改成你刚才上传的 PDF 里包含的内容！
    # 比如你的 PDF 是讲“合同法”的，你就问“违约责任是什么？”
    test_question = input("请输入测试问题:")

    answer_stream = rag.query(test_question)

    if answer_stream:
        print("\n📢 回答：")
        import time
        for chunk in answer_stream:
            content = chunk.choices[0].delta.content
            if content:
                for char in content:
                    print(char, end="", flush=True)
                    time.sleep(0.02)  #根据你的电脑性能调整大小,公式:1/你显卡(电脑)能跑的每秒token数(参考:NVDIA 5070ti and AMD 9070XT约为50token/s`; 4060以下建议调用API)
        print("\n")

