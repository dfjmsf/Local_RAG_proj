import subprocess
import sys

COMMANDS = [
    # 1. 用 Python 3.11 创建新的虚拟环境（覆盖旧的）
    ['py', '-3.11', '-m', 'venv', '.venv', '--clear'],
    # 2. 用新虚拟环境的 pip 安装依赖
    ['.venv\\Scripts\\pip.exe', 'install', '-r', 'requirements.txt'],
]

def run():
    for i, cmd in enumerate(COMMANDS):
        print(f"\n[{i+1}/{len(COMMANDS)}] 执行: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"命令失败，退出码: {result.returncode}")
            return result.returncode
    print("\n✅ 虚拟环境创建完成！Python 版本:")
    subprocess.run(['.venv\\Scripts\\python.exe', '--version'])
    return 0

if __name__ == "__main__":
    sys.exit(run())
