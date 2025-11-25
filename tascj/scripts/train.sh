#!/bin/bash
# tascj/scripts/train.sh

# 遇到任何错误立即停止执行
set -e

# ======================================================
# 1. 自动定位项目根目录
# ======================================================
# 获取当前脚本 (train.sh) 所在的绝对路径目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 假设结构是 project/tascj/scripts/train.sh
# 也就是回退两层目录到达 project_root
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "📂 Project Root detected at: ${PROJECT_ROOT}"

# ======================================================
# 2. 默认参数设置
# ======================================================
# 默认配置文件路径 (相对于项目根目录)
DEFAULT_CONFIG="config/tascj/x_2.yaml"
CONFIG_PATH="${1:-$DEFAULT_CONFIG}"

# 默认输出目录 (相对于项目根目录)
OUTPUT_ROOT="artifacts"

# ======================================================
# 3. 准备执行
# ======================================================
# 切换到项目根目录，这样 python 的导入和相对路径更清晰
cd "${PROJECT_ROOT}"

# 添加 PYTHONPATH，确保 tascj/src 可以被当作模块导入
export PYTHONPATH="${PROJECT_ROOT}/tascj/src:$PYTHONPATH"

echo "🚀 Starting Training..."
echo "   Config: ${CONFIG_PATH}"
echo "   Output: ${OUTPUT_ROOT}"

# ======================================================
# 4. 运行 Python 脚本
# ======================================================
# "${@:2}" 表示将脚本接收到的第2个及之后的所有参数传递给 python 脚本
# 例如：bash tascj/scripts/train.sh config/my.yaml --seed 42 --eval-only

python tascj/src/train.py \
    --config "${CONFIG_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    "${@:2}"