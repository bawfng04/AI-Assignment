#!/bin/bash
# ==============================================================================
# SLURM Batch Script cho D3QN Atari trên Server trường
# ==============================================================================
# SBATCH --job-name=d3qn_atari          # Tên job
# SBATCH --output=logs/d3qn_%j.out      # File lưu log output (%j là job ID)
# SBATCH --error=logs/d3qn_%j.err       # File lưu log lỗi
# SBATCH --partition=gpu                # Partition (tùy cấu hình server trường, ví dụ: gpu, rtx, etc.)
# SBATCH --gres=gpu:1                   # Yêu cầu 1 GPU (H100 20GB)
# SBATCH --cpus-per-task=4              # Số lượng CPU core (Atari cần CPU để chạy env)
# SBATCH --mem=16G                      # RAM yêu cầu
# SBATCH --time=48:00:00                # Thời gian chạy tối đa (HH:MM:SS)

echo "Job bắt đầu chạy trên node: $SLURM_JOB_NODELIST"
echo "Bắt đầu lúc: $(date)"

# Di chuyển vào thư mục project (thay đổi đường dẫn này theo server của bạn)
# cd /path/to/your/project/HW4

# Load module Python (tùy thuộc vào hệ thống module của trường)
# module load python/3.10
# module load cuda/11.8

# Kích hoạt môi trường ảo (nếu đã tạo trước bằng run_cluster.sh)
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Lỗi: Không tìm thấy thư mục .venv. Hãy chạy 'bash run_cluster.sh' trên node login trước để cài đặt môi trường."
    exit 1
fi

# Chạy mã huấn luyện
python train.py --device cuda

echo "Hoàn thành lúc: $(date)"
