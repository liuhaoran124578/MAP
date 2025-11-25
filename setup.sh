#!/bin/bash
# setup.sh

# 遇到错误立即停止
set -e

echo "🚀 Starting Environment Setup..."

# 进入脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "🌐 Setting pip mirror to Tsinghua..."
python -m pip install --upgrade pip
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️  Warning: requirements.txt not found!"
fi

echo "📦 Installing liger-kernel..."
pip install --no-deps liger-kernel==0.6.2


echo "⚙️  Installing custom OffloadAdam optimizer..."


pip install --no-deps git+https://github.com/tascj/offload_adam.git@1d0efb1d925f9a703cf66b4fbafaf904de894ebc
