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
    # print(f"1.正在扫描并加载文档... (目录: {DOCS_DIR})")
    # # --- 优化后的删除逻辑 ---
    # if os.path.exists(DB_DIR):
    #     try:
    #         shutil.rmtree(DB_DIR)
    #         print(f"已清理旧数据库: {DB_DIR}")
    #     except Exception as e:
    #         print(f"⚠️ 初次删除失败: {e}，正在重试...")
    #         time.sleep(1)
    #         try:
    #             shutil.rmtree(DB_DIR)
    #             print(f"重试删除成功！")
    #         except Exception as e2:
    #             # 如果还不行，那就真的是被占用了
    #             return False, f"无法删除旧数据库，请确保没有其他程序（如其他 Python 窗口）在使用它。\n错误信息: {e2}"
    #
    documents = load_documents(DOCS_DIR)

    if not documents:
        return False, "data/docs 文件夹为空，或没有支持的文档格式。"

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
            documents=splits,
            embedding=embedding_model,
            persist_directory=DB_DIR
        )
        return True, f"成功！共处理 {len(documents)} 份文档，生成 {len(splits)} 个向量片段。"
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