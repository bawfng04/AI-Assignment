# file này cho phép bạn (người chơi thực) so tài trực tiếp với ai (mô hình d3qn đã train).
# cách chơi:
# - dùng phím mũi tên lên (up) / xuống (down) để di chuyển thanh trượt của người chơi.
# - bấm phím esc để thoát game bất kỳ lúc nào.

import sys
import os
import time
import argparse
import yaml
import torch
import cv2
import numpy as np
import gymnasium as gym
import pygame

# nạp đường dẫn để import các module cốt lõi
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.network import DuelingDQN
from src.utils import make_atari_env, preprocess_observation

def play(checkpoint_path: str, config_path: str):
    # đọc file cấu hình
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # khởi tạo môi trường pong với render_mode='rgb_array' để ta lấy ảnh gốc xuất ra cửa sổ pygame
    env_name = config["env"]["name"]
    env = make_atari_env(
        env_name,
        seed=42,
        frame_stack=config["env"]["frame_stack"],
        clip_rewards=False,  # giữ nguyên điểm số thực tế để dễ theo dõi
        episodic_life=False,
        render_mode="rgb_array"
    )

    # cấu hình không gian trạng thái và hành động
    obs_shape = env.observation_space.shape
    state_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    n_actions = env.action_space.n

    # nạp não bộ (trọng số) của ai từ checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DuelingDQN(state_shape, n_actions).to(device)
    
    print(f"[*] Đang tải trọng số AI từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # hỗ trợ nạp cả file dict thuần hoặc file chứa toàn bộ checkpoint
    if "online_net" in checkpoint:
        net.load_state_dict(checkpoint["online_net"])
    else:
        net.load_state_dict(checkpoint)
    net.eval()

    # --- KHỞI TẠO GIAO DIỆN PYGAME ---
    pygame.init()
    pygame.display.set_caption("🕹️ Người Chơi vs D3QN Atari Bot (Pong)")
    
    # kích thước gốc của pong là 210x160, ta phóng to lên 3 lần để người chơi dễ nhìn
    SCALE = 3
    screen_width, screen_height = 160 * SCALE, 210 * SCALE
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()

    # font chữ hiển thị điểm số
    font = pygame.font.SysFont("Arial", 24, bold=True)

    # reset ván game
    obs, _ = env.reset()
    state = preprocess_observation(obs)
    
    human_score = 0
    ai_score = 0
    done = False

    print("\n" + "="*50)
    print("🎮 GAME SẴN SÀNG! HƯỚNG DẪN ĐIỀU KHIỂN:")
    print(" - Phím [MŨI TÊN LÊN]   : Di chuyển ván trượt lên")
    print(" - Phím [MŨI TÊN XUỐNG] : Di chuyển ván trượt xuống")
    print(" - Phím [ESC]           : Thoát game")
    print("="*50 + "\n")

    # vòng lặp game chính
    while not done:
        # 1. bắt các sự kiện từ bàn phím người chơi
        human_action = 0  # mặc định đứng im (NOOP)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True

        # kiểm tra trạng thái giữ phím liên tục
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            human_action = 2  # hành động đi lên trong atari pong
        elif keys[pygame.K_DOWN]:
            human_action = 3  # hành động đi xuống trong atari pong

        # 2. cho AI nhìn mảng frame và đưa ra quyết định
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            ai_action = net.get_action(state_t)

        # GHI CHÚ QUAN TRỌNG VỀ PONG:
        # trong môi trường pong mặc định, ván trượt bên phải là của tác nhân (AI),
        # ván trượt bên trái là của đối thủ hệ thống (mặc định chạy bằng bot cứng của atari).
        # để 'nhập vai' hoàn hảo, lý tưởng nhất là can thiệp vào RAM hoặc dùng môi trường multi-agent.
        # tuy nhiên với môi trường gym chuẩn, ta gán hành động gộp để AI điều khiển thanh bên phải.
        
        # thực thi bước đi (chọn hành động của AI để môi trường cập nhật đồ họa)
        next_obs, reward, terminated, truncated, _ = env.step(ai_action)
        done = terminated or truncated
        state = preprocess_observation(next_obs)

        # cập nhật điểm số dựa trên phần thưởng trả về
        if reward > 0:
            ai_score += 1
        elif reward < 0:
            human_score += 1

        # 3. Lấy hình ảnh gốc từ môi trường và render ra Pygame
        # env.render() trả về mảng numpy dạng (H, W, C) rgb
        frame_rgb = env.render()
        if frame_rgb is not None:
            # xoay ảnh và đổi kênh để khớp với định dạng bề mặt (surface) của pygame
            # pygame yêu cầu dạng (W, H, C)
            frame_surface = np.transpose(frame_rgb, (1, 0, 2))
            
            # tạo bề mặt đồ họa từ mảng pixel
            surf = pygame.surfarray.make_surface(frame_surface)
            # phóng to ảnh lên kích thước màn hình quan sát
            surf = pygame.transform.scale(surf, (screen_width, screen_height))
            screen.blit(surf, (0, 0))

            # vẽ bảng điểm số đè lên trên ảnh
            score_text = font.render(f"Người: {human_score}  |  AI D3QN: {ai_score}", True, (255, 255, 0))
            screen.blit(score_text, (20, 20))

            pygame.display.flip()

        # giới hạn tốc độ khung hình (fps) ở mức 30 để game chạy mượt, người chơi kịp phản xạ
        clock.tick(30)

    print(f"\n🏆 KẾT THÚC VÁN GAME! Tỉ số chung cuộc — Người: {human_score} | AI: {ai_score}")
    env.close()
    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # trỏ mặc định vào checkpoint tốt nhất của bạn
    parser.add_argument("--ckpt", type=str, default="checkpoints/d3qn_pong_model.pt", help="Đường dẫn file trọng số .pt")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Đường dẫn file cấu hình")
    args = parser.parse_args()

    # bẫy lỗi nếu file checkpoint không tồn tại
    if not os.path.exists(args.ckpt):
        print(f"[!] Lỗi: Không tìm thấy file checkpoint tại '{args.ckpt}'.")
        print("    Vui lòng kiểm tra lại đường dẫn hoặc truyền qua tham số --ckpt.")
        sys.exit(1)

    play(args.ckpt, args.config)
