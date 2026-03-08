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
PARENT_MAP_PATH = os.path.join(DB_DIR, "parent_map.json")


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

        # D. 加载父文档映射表 (用于通过 parent_id 还原父文档内容)
        self.parent_map = {}
        if os.path.exists(PARENT_MAP_PATH):
            try:
                import json as _json
                with open(PARENT_MAP_PATH, "r", encoding="utf-8") as f:
                    self.parent_map = _json.load(f)
                print(f" -> 已加载父文档映射: {len(self.parent_map)} 条")
            except Exception as e:
                print(f"⚠️ 加载父文档映射失败: {e}")

        print("✅ 系统初始化完成！")

    # [新增] 获取已索引的文件列表
    def get_indexed_files(self):
        """
        查询 ChromaDB，返回当前数据库中所有唯一的 source 文件名 (纯文件名，不带路径)
        """
        try:
            # 仅获取 metadatas，速度快
            data = self.vector_db.get(include=['metadatas'])

            if not data or not data['metadatas']:
                return set()

            indexed_files = set()
            for meta in data['metadatas']:
                if meta and 'source' in meta:
                    # [关键修复] 无论数据库里存的是绝对路径还是相对路径
                    # 这里统一只提取文件名，比如 "C:/data/docs/a.txt" -> "a.txt"
                    # 这样才能和 os.listdir 的结果进行比对
                    full_path = meta['source']
                    file_name = os.path.basename(full_path)
                    indexed_files.add(file_name)

            return indexed_files
        except Exception as e:
            print(f"❌ 获取索引文件列表失败: {e}")
            return set()

    def route_query(self, question):
        """
        判断用户意图：是需要检索(SEARCH)还是闲聊(CHAT)
        """
        print(f"🚦 正在进行意图路由分析: {question}")

        # 极简 Prompt，强制模型只输出关键词
        # 极简 Prompt，强制模型只输出关键词
        system_prompt = (
            "You are a routing system. Analyze the user's question. "
            "If the question implies looking up specific documents, facts, or context, output 'SEARCH'. "
            "If the question is a greeting, general knowledge, coding request, or translation, output 'CHAT'. "
            "Output ONLY 'SEARCH' or 'CHAT'. Do not explain."
        )

        try:
            url = "http://127.0.0.1:1234/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            data = {
                "model" : "local-model",
                "messages" : [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                "temperature" : 0.0,
                "max_tokens" : 1000,
                "stream" : False
            }

            response = requests.post(
                url, headers = headers , json = data,
                proxies = {"http" : None, "https" : None}, timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                raw_content = result['choices'][0]['message']['content'].strip()

                # [新增] 调试打印：看看模型到底输出了什么妖魔鬼怪
                print(f"   [Debug] 路由原始输出: {raw_content}")

                # [修改 2] 清洗 <think> 标签
                # DeepSeek-R1 喜欢输出 <think>思考过程</think> SEARCH
                final_intent = raw_content
                if "</think>" in raw_content:
                    # 只取 </think> 后面的部分
                    final_intent = raw_content.split("</think>")[-1].strip()

                final_intent = final_intent.upper()

                # [修改 3] 判定逻辑
                # 只要包含 CHAT 就认为是闲聊，否则默认 SEARCH (更安全的策略)
                if "CHAT" in final_intent:
                    return "CHAT"

                return "SEARCH"

            print(f"❌ 路由API报错: {response.status_code}")
            return "SEARCH"  # 失败默认走搜索

        except Exception as e:
            print(f"❌ 路由失败: {e}，默认走 SEARCH")
            return "SEARCH"

    #  抽离出的 LLM 调用通用函数
    def _call_llm(self, messages):
        print("\n🤖 DeepSeek 正在思考...")
        url = "http://127.0.0.1:1234/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "local-model",
            "messages": messages,
            "temperature": 0.3,
            "stream": True
        }
        try:
            response = requests.post(
                url, headers=headers, json=data, stream=True,
                proxies={"http": None, "https": None}, timeout=60
            )
            if response.status_code != 200:
                print(f"❌ 服务器返回错误: {response.status_code}")
                print(response.text)
                return None
            return response
        except Exception as e:
            print(f"❌ LLM 调用失败: {e}")
            return None


    def query(self, question, history=[], mode="flash", use_reranker=False):
        """
        :param question: 用户问题
        :param mode: 'flash' (极速) 或 'pro' (深度)
        :param history: 前端传来的历史对话列表 (list of dict)
        :return: (response对象, 参考文档列表)
        """

        # 1. [新增] 执行路由判断
        intent = self.route_query(question)
        print(f"👉 路由结果: {intent}")

        print(f"\n🔍 正在检索：{question} | 模式: {mode.upper()}")

        final_docs = []
        search_query = question

        # === 分支 A: 闲聊模式 (CHAT) ===
        if intent == "CHAT":
            # 直接构建 Prompt，不查库
            print("💬 进入闲聊模式，跳过检索...")

            # 使用更通用的 System Prompt
            system_prompt = "你是一个乐于助人的 AI 助手。请直接回答用户的问题。"
            messages_payload = [{"role": "system", "content": system_prompt}]
            # 注入历史
            if history:
                messages_payload.extend(history[-6:])
            messages_payload.append({"role": "user", "content": question})

            # 直接调用 LLM
            response = self._call_llm(messages_payload)
            # 返回时 doc 列表为空，前端就不会显示“参考来源”
            return response, [], intent  # 把 intent 也返回给前端用于展示

        # === 分支 B: 检索模式 (SEARCH) ===
        else: # intent == "SEARCH"
            print("🔍 进入检索模式...")

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

                    # 4. 排序截断 (Top-5)
                    # 将文档和分数打包，按分数降序排
                    scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)

                    print("\n📊 Rerank 打分结果 (Top-5):")
                    for doc, score in scored_docs[:5]:
                        print(f"   [分: {score:.4f}] {doc.page_content[:30]}...")

                    # 取前 5 名的文档对象
                    final_docs = [doc for doc, score in scored_docs[:5]]
                else:
                    print("⚠️ 初步检索未找到文档。")

            else:
                # === Flash 模式 (极速) ===
                # 直接找 Top-5，不经过模型重算，速度最快
                final_docs = self.vector_db.similarity_search(question, k= 5)


            # --- 通用逻辑 ---
            if not final_docs:
                print("⚠️ 未找到相关文档。")
                return None, [], intent

            print("\n📚 最终参考资料 (Parent-Child 还原)：")
            context_text = ""
            used_parents = set() # 用于去重，防止多个子块属于同一个父块，导致重复阅读

            for i, doc in enumerate(final_docs):
                # [关键] 通过 parent_id 从映射表还原父文档内容
                # 兼容旧数据：如果没有 parent_id，回退使用子文档自身的内容
                parent_id = doc.metadata.get("parent_id", "")
                if parent_id and parent_id in self.parent_map:
                    content = self.parent_map[parent_id]
                else:
                    # 旧版数据库可能存了 parent_content，做兼容
                    content = doc.metadata.get("parent_content", doc.page_content)

                # [去重逻辑]
                # 计算内容的哈希值或直接用字符串判断，防止重复添加相同的父文档
                # 这里简单用字符串长度+前100字作为简易指纹
                content_fingerprint = f"{len(content)}_{content[:50]}"

                if content_fingerprint in used_parents:
                    print(f"   [跳过] 子块 {i+1} 指向已存在的父块...")
                    continue

                used_parents.add(content_fingerprint)

                # 打印预览 (预览一下子块的来源)
                source = os.path.basename(doc.metadata.get("source", "unknown"))
                cleaned_content = content[:50].replace('\n', '')
                print(f"[{len(used_parents)}] 来源: {source} | 内容预览: {cleaned_content}...")
                # 拼接到 Context
                context_text += f"片段{len(used_parents)}: {content}\n\n"

        # --- 步骤 2: 构建 Prompt 与 历史消息注入 ---
        # 1. 定义系统提示词 (Persona)
        system_prompt = "你是一个专业助手。请根据【参考资料】回答问题。如果不知道就说不知道。"

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
        response = self._call_llm(messages_payload)

        # 返回 3 个值: 响应流, 文档列表, 意图
        return response, final_docs, intent

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