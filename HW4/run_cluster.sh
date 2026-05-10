#!/bin/bash
# ==============================================================================
# Script setup môi trường và chạy train trên Server/Cluster
# ==============================================================================
# Lưu ý: Chạy script này bằng lệnh: bash run_cluster.sh

# Dừng script nếu có lỗi xảy ra
set -e

echo "============================================================"
echo "  Bắt đầu setup môi trường cho D3QN Atari"
echo "============================================================"

# 1. Khởi tạo môi trường ảo (Virtual Environment) với Python 3.10 hoặc 3.11
# (Tránh dùng Python 3.12+ hoặc 3.14 vì ale-py hiện chưa có pre-built wheels)
echo "=> Đang tạo virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    # Thử dùng python3, python3.11 hoặc python3.10 tùy hệ thống
    if command -v python3.11 &> /dev/null; then
        python3.11 -m venv .venv
    elif command -v python3.10 &> /dev/null; then
        python3.10 -m venv .venv
    else
        python3 -m venv .venv
    fi
fi

# Kích hoạt môi trường ảo
echo "=> Kích hoạt virtual environment..."
source .venv/bin/activate

# 2. Cập nhật pip và cài đặt dependencies
echo "=> Cài đặt/cập nhật pip..."
pip install --upgrade pip

echo "=> Cài đặt thư viện từ requirements.txt..."
pip install -r requirements.txt

# 3. Chạy huấn luyện
echo "============================================================"
echo "  Bắt đầu quá trình huấn luyện (Training)"
echo "============================================================"

# Tạo thư mục lưu log và checkpoint nếu chưa có
mkdir -p checkpoints runs

# Cấu hình tham số môi trường CUDA để đảm bảo nhận đúng GPU
# GPU H100 20GB dư sức chạy model này, ta thiết lập device mặc định là cuda
export CUDA_VISIBLE_DEVICES=0

# Chạy train
# (Bạn có thể thêm --env BreakoutNoFrameskip-v4 nếu muốn chạy game khác)
# (Thêm nohup ... & ở cuối nếu muốn chạy ngầm khi đóng terminal)
python train.py --device cuda

echo "============================================================"
echo "  Hoàn thành chạy script!"
echo "============================================================"
