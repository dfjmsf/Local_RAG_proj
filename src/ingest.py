import os
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
    print(f"1.正在扫描并加载文档... (目录: {DOCS_DIR})")
    documents = load_documents(DOCS_DIR)

    if not documents:
        print("❌ 没有加载到任何文档。")
        return

    print(f"   -> 共成功加载 {len(documents)} 个文档片段。")


    print("2. 正在切分文本...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,   # chunk_size=500: 每块文本大约 500 个字符。太小了语义不全，太大了检索不准。
        chunk_overlap=50  # chunk_overlap=50: 相邻两块文本有 50 字重叠，防止关键信息刚好被切断。
    )
    splits = text_splitter.split_documents(documents)

    print(f"   -> 文档被切分成了 {len(splits)} 个片段 (Chunks)。")

    print("3. 正在初始化向量模型 (首次运行会自动下载模型，请耐心等待)...")
    # 使用 sentence-transformers 的经典模型 'all-MiniLM-L6-v2'
    # 这个模型很小(约80MB)，速度快，效果好
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # 强制用 CPU，避免和 LM Studio 抢显存
    )

    print("4. 正在将数据存入向量数据库 ChromaDB...")
    # 这一步会做两件事：
    # 1. 把所有文本片段扔给 embedding_model 变成向量
    # 2. 把向量和原始文本存入本地文件夹 DB_DIR
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )

    print(f"✅ 成功！向量数据库已构建完成。存储位置: {DB_DIR}")
    print(f"   共存储了 {len(splits)} 条向量数据。")

if __name__ == '__main__':
    create_vector_db()