"""
RAG 核心引擎 (rag_core02.py)
功能: 意图路由 → 混合检索 (向量 + BM25 + RRF) → Reranker 精排 → LLM 流式调用
依赖: LM Studio (本地 DeepSeek-R1), ChromaDB, Sentence-Transformers, BM25
"""

import os
import re
import json
import requests
import jieba
from rank_bm25 import BM25Okapi

# 强制离线模式 (禁止 HuggingFace 联网下载)
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# ============================================================
# 路径配置
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, "../data/chroma_db")
RERANK_MODEL_PATH = os.path.join(CURRENT_DIR, "../model_cache/bge-reranker-base")
PARENT_MAP_PATH = os.path.join(DB_DIR, "parent_map.json")

# LM Studio API
LLM_URL = "http://127.0.0.1:1234/v1/chat/completions"
LLM_HEADERS = {"Content-Type": "application/json"}
LLM_PROXIES = {"http": None, "https": None}  # 绕过系统代理


class RAGSystem:
    """本地化 RAG 系统：混合检索 + Reranker + 意图路由"""

    # ============================================================
    # 初始化
    # ============================================================

    def __init__(self):
        print("正在初始化 RAG 系统...")

        # A. 向量 Embedding 模型
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

        # C. Reranker 精排模型 (可选, 加载失败自动降级)
        print(f" -> 正在加载 Rerank 模型 ({RERANK_MODEL_PATH})...")
        try:
            self.reranker = CrossEncoder(RERANK_MODEL_PATH, device="cpu")
            print(" -> Rerank 模型加载成功！")
        except Exception as e:
            print(f"❌ Rerank 模型加载失败: {e}")
            print("   (将自动降级为仅使用向量检索)")
            self.reranker = None

        # D. 父文档映射表 (parent_id → 父文档内容)
        self.parent_map = {}
        if os.path.exists(PARENT_MAP_PATH):
            try:
                with open(PARENT_MAP_PATH, "r", encoding="utf-8") as f:
                    self.parent_map = json.load(f)
                print(f" -> 已加载父文档映射: {len(self.parent_map)} 条")
            except Exception as e:
                print(f"⚠️ 加载父文档映射失败: {e}")

        # E. BM25 索引 (混合检索)
        self._build_bm25_index()

        print("✅ 系统初始化完成！")

    # ============================================================
    # 文件索引查询
    # ============================================================

    def get_indexed_files(self):
        """查询 ChromaDB，返回当前数据库中所有唯一的 source 文件名"""
        try:
            data = self.vector_db.get(include=['metadatas'])
            if not data or not data['metadatas']:
                return set()

            indexed_files = set()
            for meta in data['metadatas']:
                if meta and 'source' in meta:
                    indexed_files.add(os.path.basename(meta['source']))
            return indexed_files
        except Exception as e:
            print(f"❌ 获取索引文件列表失败: {e}")
            return set()

    # ============================================================
    # BM25 混合检索
    # ============================================================

    def _build_bm25_index(self):
        """从 ChromaDB 中加载全部文档，用 jieba 分词后构建 BM25 索引"""
        print(" -> 正在构建 BM25 索引...")
        try:
            data = self.vector_db.get(include=['documents', 'metadatas'])

            if not data or not data['documents']:
                print("⚠️ 数据库为空，BM25 索引跳过")
                self.bm25_index = None
                self.bm25_docs = []
                self.bm25_metadatas = []
                self.bm25_corpus = []
                return

            self.bm25_docs = data['documents']
            self.bm25_metadatas = data['metadatas']
            self.bm25_corpus = [list(jieba.cut(doc)) for doc in self.bm25_docs]
            self.bm25_index = BM25Okapi(self.bm25_corpus)
            print(f" -> BM25 索引构建完成！共 {len(self.bm25_docs)} 个文档片段")
        except Exception as e:
            print(f"⚠️ BM25 索引构建失败: {e}")
            self.bm25_index = None
            self.bm25_docs = []
            self.bm25_metadatas = []
            self.bm25_corpus = []

    def _bm25_search(self, query, k=10):
        """BM25 关键词检索，返回 Top-K 的 (文档内容, metadata, 索引) 列表"""
        if not self.bm25_index:
            return []

        query_tokens = list(jieba.cut(query))
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [
            (self.bm25_docs[idx], self.bm25_metadatas[idx], idx)
            for idx in top_indices if scores[idx] > 0
        ]

    def _hybrid_search(self, query, k=5):
        """
        混合检索：向量检索 + BM25 → RRF (Reciprocal Rank Fusion) 融合
        返回 LangChain Document 对象列表
        """
        RRF_K = 60

        # 路径 1: 向量语义检索
        vector_k = min(k * 3, 20)
        vector_docs = self.vector_db.similarity_search(query, k=vector_k)

        # 路径 2: BM25 关键词检索
        bm25_results = self._bm25_search(query, k=vector_k)

        # RRF 融合
        rrf_scores = {}
        doc_map = {}
        vec_hashes = set()
        bm25_hashes = set()

        for rank, doc in enumerate(vector_docs):
            h = hash(doc.page_content)
            rrf_scores[h] = rrf_scores.get(h, 0) + 1.0 / (RRF_K + rank + 1)
            doc_map[h] = doc
            vec_hashes.add(h)

        for rank, (content, metadata, _) in enumerate(bm25_results):
            h = hash(content)
            rrf_scores[h] = rrf_scores.get(h, 0) + 1.0 / (RRF_K + rank + 1)
            bm25_hashes.add(h)
            if h not in doc_map:
                doc_map[h] = Document(page_content=content, metadata=metadata or {})

        sorted_hashes = sorted(rrf_scores.keys(), key=lambda h: rrf_scores[h], reverse=True)[:k]
        final_docs = [doc_map[h] for h in sorted_hashes]

        # ========== 检索质量指标 ==========
        overlap = vec_hashes & bm25_hashes
        all_unique = vec_hashes | bm25_hashes
        overlap_rate = len(overlap) / len(all_unique) * 100 if all_unique else 0
        top_rrf = [rrf_scores[h] for h in sorted_hashes]

        print(f"\n   {'='*50}")
        print(f"   📊 检索质量报告")
        print(f"   {'='*50}")
        print(f"   │ 向量检索: {len(vector_docs)}条  |  BM25检索: {len(bm25_results)}条")
        print(f"   │ 去重后独立文档: {len(all_unique)}条  |  双路重合: {len(overlap)}条")
        print(f"   │ 🎯 双路重合率: {overlap_rate:.1f}%")
        if overlap_rate > 50:
            print(f"   │    → 重合率高，语义与关键词高度一致，检索置信度高")
        elif overlap_rate > 20:
            print(f"   │    → 重合率中等，混合检索有效互补")
        else:
            print(f"   │    → 重合率低，两路检索差异大，混合召回提升明显")
        print(f"   │ RRF 分数分布 (Top-{len(top_rrf)}):")
        for i, score in enumerate(top_rrf):
            bar = '█' * int(score * 2000)
            print(f"   │   #{i+1}: {score:.5f} {bar}")
        print(f"   {'='*50}\n")

        return final_docs

    # ============================================================
    # 意图路由
    # ============================================================

    def route_query(self, question):
        """判断用户意图：SEARCH (检索知识库) 或 CHAT (闲聊)"""
        print(f"🚦 正在进行意图路由分析: {question}")

        # 注入知识库文件名，让模型了解知识库内容
        indexed_files = self.get_indexed_files()
        if indexed_files:
            file_list = ", ".join(sorted(indexed_files))
            kb_context = f"\nThe knowledge base currently contains: [{file_list}]. "
        else:
            kb_context = "\nThe knowledge base is currently empty. "

        system_prompt = (
            "You are an intent routing AI. Your task is to ONLY output 'SEARCH' or 'CHAT'. Skip thinking process.\n"
            f"{kb_context}\n"
            "INSTRUCTIONS:\n"
            "1. Carefully compare the user's question with the provided filenames to see if they relate.\n"
            "2. If related to ANY filename, output 'SEARCH'.\n"
            "3. If question is a greeting, general coding, or unrelated, output 'CHAT'.\n\n"
            "EXAMPLES:\n"
            "User: '你好' -> Output: CHAT\n"
            "User: '帮我写个Python脚本' -> Output: CHAT\n"
            "User: '根据文档，xxx是什么？' -> Output: SEARCH\n"
            "Now it's your turn. Output ONLY 'SEARCH' or 'CHAT'."
        )

        try:
            data = {
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                "temperature": 0.0,
                "max_tokens": 1000,
                "stream": False
            }

            response = requests.post(
                LLM_URL, headers=LLM_HEADERS, json=data,
                proxies=LLM_PROXIES, timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                raw_content = result['choices'][0]['message']['content'].strip()
                print(f"   路由原始输出: {raw_content}")

                # 清洗 <think> 标签 (DeepSeek-R1 可能输出思考过程)
                final_intent = raw_content
                if "</think>" in raw_content:
                    final_intent = raw_content.split("</think>")[-1]

                # 正则提取 SEARCH 或 CHAT
                matches = re.findall(r'\b(SEARCH|CHAT)\b', final_intent.upper())
                if matches:
                    return matches[-1]
                return "SEARCH"  # 无法解析时默认检索

            print(f"❌ 路由 API 报错: {response.status_code}")
            return "SEARCH"

        except Exception as e:
            print(f"❌ 路由失败: {e}，默认走 SEARCH")
            return "SEARCH"

    # ============================================================
    # LLM 调用
    # ============================================================

    def _call_llm(self, messages):
        """调用 LM Studio 的 DeepSeek 模型，返回流式 response 对象"""
        print("\n🤖 DeepSeek 正在思考...")
        data = {
            "model": "local-model",
            "messages": messages,
            "temperature": 0.3,
            "stream": True
        }
        try:
            response = requests.post(
                LLM_URL, headers=LLM_HEADERS, json=data,
                stream=True, proxies=LLM_PROXIES, timeout=60
            )
            if response.status_code != 200:
                print(f"❌ 服务器返回错误: {response.status_code}")
                print(response.text)
                return None
            return response
        except Exception as e:
            print(f"❌ LLM 调用失败: {e}")
            return None

    # ============================================================
    # 主查询入口
    # ============================================================

    def query(self, question, history=None, mode="flash"):
        """
        RAG 主查询入口
        :param question: 用户问题
        :param history: 前端传来的历史对话列表 (list of dict)
        :param mode: 'flash' (极速) 或 'pro' (深度)
        :return: (response 对象, 参考文档列表, 意图)
        """
        if history is None:
            history = []

        # 1. 意图路由
        intent = self.route_query(question)
        print(f"👉 路由结果: {intent}")
        print(f"\n🔍 正在检索：{question} | 模式: {mode.upper()}")

        # === 分支 A: 闲聊模式 ===
        if intent == "CHAT":
            print("💬 进入闲聊模式，跳过检索...")
            system_prompt = "你是一个乐于助人的 AI 助手。请直接回答用户的问题。"
            messages_payload = [{"role": "system", "content": system_prompt}]
            if history:
                messages_payload.extend(history[-6:])
            messages_payload.append({"role": "user", "content": question})
            response = self._call_llm(messages_payload)
            return response, [], intent

        # === 分支 B: 检索模式 ===
        print("🔍 进入检索模式...")
        final_docs = []

        if mode == "pro" and self.reranker:
            # Pro 模式: 混合检索 Top-20 → Reranker 精排 → Top-5
            initial_docs = self._hybrid_search(question, k=20)

            if initial_docs:
                pairs = [[question, doc.page_content] for doc in initial_docs]
                print(" -> 正在进行 Rerank 重排序...")
                scores = self.reranker.predict(pairs)
                scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
                top5 = scored_docs[:5]
                top5_scores = [s for _, s in top5]

                # Reranker 质量指标
                avg_score = sum(top5_scores) / len(top5_scores) if top5_scores else 0
                max_score = max(top5_scores) if top5_scores else 0
                min_score = min(top5_scores) if top5_scores else 0

                print(f"\n   {'='*50}")
                print(f"   🏆 Reranker 质量报告")
                print(f"   {'='*50}")
                print(f"   │ 均分: {avg_score:.4f}  |  最高: {max_score:.4f}  |  最低: {min_score:.4f}")
                if avg_score > 0.5:
                    print(f"   │ 🟢 质量优秀，文档与问题高度相关")
                elif avg_score > 0:
                    print(f"   │ 🟡 质量中等，部分文档相关")
                else:
                    print(f"   │ 🔴 质量较低，知识库可能缺少相关内容")
                print(f"   │ Top-5 明细:")
                for doc, score in top5:
                    print(f"   │   [{score:+.4f}] {doc.page_content[:35]}...")
                print(f"   {'='*50}\n")

                final_docs = [doc for doc, score in top5]
            else:
                print("⚠️ 混合检索未找到文档。")
        else:
            # Flash 模式: 混合检索 Top-5
            final_docs = self._hybrid_search(question, k=5)

        # 通用逻辑: 构建上下文
        if not final_docs:
            print("⚠️ 未找到相关文档。")
            return None, [], intent

        print("\n📚 最终参考资料 (Parent-Child 还原)：")
        context_text = ""
        used_parents = set()

        for i, doc in enumerate(final_docs):
            # 通过 parent_id 还原父文档内容，兼容旧数据
            parent_id = doc.metadata.get("parent_id", "")
            if parent_id and parent_id in self.parent_map:
                content = self.parent_map[parent_id]
            else:
                content = doc.metadata.get("parent_content", doc.page_content)

            # 去重: 用内容指纹避免重复父文档
            fingerprint = f"{len(content)}_{content[:50]}"
            if fingerprint in used_parents:
                print(f"   [跳过] 子块 {i+1} 指向已存在的父块...")
                continue
            used_parents.add(fingerprint)

            source = os.path.basename(doc.metadata.get("source", "unknown"))
            preview = content[:50].replace('\n', '')
            print(f"[{len(used_parents)}] 来源: {source} | 预览: {preview}...")
            context_text += f"片段{len(used_parents)}: {content}\n\n"

        # 构建 Prompt 与历史注入
        system_prompt = "你是一个专业助手。请根据【参考资料】回答问题。如果不知道就说不知道。"
        messages_payload = [{"role": "system", "content": system_prompt}]

        if history:
            messages_payload.extend(history[-6:])
            print(f" -> 已注入历史记忆: {len(history[-6:])} 条消息")

        current_user_prompt = f"【参考资料】:\n{context_text}\n\n【问题】:\n{question}"
        messages_payload.append({"role": "user", "content": current_user_prompt})

        # 调用 LLM
        response = self._call_llm(messages_payload)
        return response, final_docs, intent