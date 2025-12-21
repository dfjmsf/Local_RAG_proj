import os
import time
import shutil
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))# 获取当前脚本所在的绝对路径，确保在任何地方运行都不会找不到文件
DOCS_DIR = os.path.join(CURRENT_DIR, '../data/docs')# 数据的输入目录
DB_DIR = os.path.join(CURRENT_DIR, "../data/chroma_db")
LOADER_MAPPING = {
    ".pdf": (PyPDFLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    ".csv": (CSVLoader, {"encoding": "utf-8"}),
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".md": (TextLoader, {"encoding": "utf-8"}),
}

def load_documents(source_dir):
    all_documents = []

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext in LOADER_MAPPING:
                loader_class, loader_arge = LOADER_MAPPING[file_ext]
                try:
                    print(f"Loading:{file}...")
                    loader = loader_class(file_path, **loader_arge)
                    documents = loader.load()
                    all_documents.extend(documents)
                except Exception as e:
                    print(f"❌ 加载文件失败 {file}: {e}")

            else:
                pass

        return all_documents
def create_vector_db():

    documents = load_documents(DOCS_DIR)

    if not documents:
        return False, "data/docs 文件夹为空，或没有支持的文档格式。"

    print(f"   -> 共成功加载 {len(documents)} 个文档片段。")


    print("2. 正在切分文本...")

    # A. 定义父切分器 (Parent Splitter) - 大块，用于给 AI 看
    # 800 字左右通常包含一个完整的段落逻辑
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,   # chunk_size=: 每块文本大约  个字符。太小了语义不全，太大了检索不准。
        chunk_overlap=0  # chunk_overlap=: 相邻两块文本有  字重叠，防止关键信息刚好被切断。
    )

    # B. 定义子切分器 (Child Splitter) - 小块，用于生成向量检索
    # 200 字左右语义最致密，检索最准

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    print("2. 正在进行父子切分...")

    # C. 执行切分流程
    # 1. 先切出父文档
    parent_docs = parent_splitter.split_documents(documents)

    final_storage_docs = []  # 最终要存入数据库的文档列表

    # 2. 遍历每个父文档，切分成子文档
    from langchain_core.documents import Document   # 确保引入 Document 对象

    for parent_doc in parent_docs:
        # 获取父文档的内容
        parent_content = parent_doc.page_content
        # 获取父文档原有的 metadata (比如 source, page)
        base_metadata = parent_doc.metadata.copy()

        # 切分子文档
        child_texts = child_splitter.split_text(parent_content)

        for child_text in child_texts:
            # [关键步骤] : 在子文档的 metadata 里存储父文档的内容
            # 这样检索到子文档时，就能顺藤摸瓜找到父文档
            new_metadata = base_metadata.copy()
            new_metadata["parent_content"] = parent_content # <--- 存入父内容

            # 创建新的子文档对象
            child_doc = Document(page_content=child_text, metadata=new_metadata)
            final_storage_docs.append(child_doc)

    print(f"   -> 父文档数: {len(parent_docs)} | 子文档数 (实际入库): {len(final_storage_docs)}")

    # 使用 sentence-transformers 的经典模型 'all-MiniLM-L6-v2'
    # 这个模型很小(约80MB)，速度快，效果好
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}  # 强制用 CPU，避免和 LM Studio 抢显存
        )

        print("正在连接数据库...")
        vectordb = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding_model
        )

        try:
            print("正在清空旧数据...")
            # 这里的逻辑是：删除整个集合，然后 LangChain 会在下面重新创建它
            vectordb.delete_collection()
        except Exception:
            # 如果第一次运行，集合可能不存在，报错也没关系
            pass

        print("正在写入新数据...")
        vectordb = Chroma.from_documents(
            documents=final_storage_docs,
            embedding=embedding_model,
            persist_directory=DB_DIR
        )
        return True, f"成功！采用父子索引策略。生成 {len(final_storage_docs)} 个子向量片段。"
    except Exception as e:
        return False, f"向量库构建失败: {e}"

def reset_vector_db():
    """
        独立功能：清空向量数据库，但不重新构建。
        用于"清空所有"按钮。
    """
    try:
        # 1. 初始化 Embedding (连接数据库需要它)
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        # 2. 连接到数据库
        print("正在连接数据库以进行重置...")
        vectordb = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding_model
        )

        # 3. 删除集合 (逻辑清空)
        try:
            vectordb.delete_collection()
            print()
            print("数据库集合已删除。")
            return True, "数据库已重置为空。"
        except  ValueError:
            # 如果数据库本来就是空的，delete_collection 会报错，但这不算失败
            return True, "数据库本来就是空的。"

    except Exception as e:
        return False, f"重置数据库失败: {e}"

if __name__ == '__main__':
    success, msg = create_vector_db()
    print(msg)