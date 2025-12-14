import sys
import os
import time
# --- 关键修改：强制禁用本地连接的代理 ---
# 这告诉 Python：去 localhost 或 127.0.0.1 的路，直接走，不要经过 VPN
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
# 为了保险，把 HTTP_PROXY 临时清空（只影响当前脚本运行，不影响系统）
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

print(f"正在连接本地LM Studio(Python { sys.version.split()[0] })....")

try:

    history = [
        {"role": "system", "content": "你是一个乐于助人的AI助手"},
        {"role": "user", "content": "你好！请用一句话解释什么是'向量数据库'"}
    ]

    completion = client.chat.completions.create(
        model="local-model",
        messages=history,
        temperature=0.7,
        stream=True
    )

    print("\nDeepSeek 回答:")
    print("-" * 30)

    for chunk in completion:
        content = chunk.choices[0].delta.content

        if content:
            for char in content:
                print(char, end="" ,flush=True)
                time.sleep(0.02)

    print("\n" + "-" * 30)
    print("\n测试成功,连接通畅")

except Exception as e:
    print(f"\n❌ 连接失败: {e}")
    print("请检查：\n1. LM Studio 的 Server 是否已点击 'Start'?\n2. 端口是否是 1234?")
