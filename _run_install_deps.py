import subprocess
import sys

def run():
    result = subprocess.run(
        ['.venv\\Scripts\\pip.exe', 'install', '-r', 'requirements.txt'],
        check=False,
    )
    return result.returncode

if __name__ == "__main__":
    sys.exit(run())
