import os

# 临时允许联网下载 (覆盖掉系统环境变量里的离线设置)
os.environ["HF_HUB_OFFLINE"] = "0"
#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # (可选) 使用国内镜像加速下载

# 3. 这里的设置取决于你的 VPN 软件
# 如果你开了 VPN，通常不需要手动设代理，Python 会自动走系统代理。
# 但为了保险，可以尝试不设任何代理变量，或者根据你的 VPN 端口设置（比如 7890）
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897" # <--- 仅作为备选参考

from sentence_transformers import CrossEncoder


def download_reranker():
    model_name = "BAAI/bge-reranker-base"
    print(f"📡 正在连接 HuggingFace 下载模型: {model_name}")
    print("⏳ 文件较大 (约 500MB - 1GB)，请耐心等待...")

    try:
        # 这行代码会自动下载模型并存入本地缓存目录
        # (默认在 C:\Users\你的用户名\.cache\huggingface\...)
        model = CrossEncoder(model_name)
        print(f"✅ 下载成功！模型已缓存到本地。")

        # 简单测试一下，确保模型能跑
        scores = model.predict([('Query', 'Paragraph')])
        print("✅ 模型加载测试通过！")

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("💡 建议：如果你有 VPN，请开启全局代理；或者检查网络连接。")


if __name__ == "__main__":
    download_reranker()