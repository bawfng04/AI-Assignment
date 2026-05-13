# file này tạo ra một ván game pong đồ họa tốc độ cực chậm để demo trực quan.
# TRONG MÔI TRƯỜNG ATARI PONG GỐC:
# - Thanh bên TRÁI mặc định được điều khiển bởi bot máy của Atari.
# - Thanh bên PHẢI là đối tượng duy nhất nhận lệnh từ hàm env.step().
#
# CƠ CHẾ ĐIỀU KHIỂN HYBRID CỰC HAY CHO MÀN DEMO NÀY:
# - Nếu bạn BẤM GIỮ phím mũi tên Lên/Xuống: Bạn (Người chơi) sẽ giành quyền điều khiển thanh bên phải.
# - Nếu bạn BUÔNG TAY (Không bấm phím): Não bộ AI (chạy ngầm bằng giải thuật Heuristic Computer Vision)
#   sẽ tự động tiếp quản thanh bên phải để biểu diễn những pha cứu bóng xuất thần!

import sys
import os
import time
import argparse
import numpy as np
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import pygame

def play():
    # --- KHỞI TẠO MÔI TRƯỜNG GỐC (KHÔNG BỊ LẶP FRAME TỐC ĐỘ CAO) ---
    # ta gọi trực tiếp gym.make thay vì make_atari_env để tránh lớp MaxAndSkipEnv (skip=4).
    # việc bỏ skip frame giúp game chạy chính xác từng khung hình 1, tốc độ cực kỳ chậm và êm ái.
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.metadata["render_fps"] = 30

    # --- KHỞI TẠO ĐỒ HỌA PYGAME ---
    pygame.init()
    pygame.display.set_caption("Nguoi Choi / AI Hybrid vs Atari System (Pong)")
    
    # Kích thước gốc của Pong là 160x210, phóng to lên 3.5 lần cho hoành tráng
    SCALE = 3.5
    screen_width, screen_height = int(160 * SCALE), int(210 * SCALE)
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 22, bold=True)

    env.reset()
    
    right_score = 0
    left_score = 0
    done = False

    print("\n" + "="*55)
    print(" GIAO DIEN GAME CHAM & DIEU KHIEN HYBRID SAN SANG!")
    print(" - Phim [MUI TEN LEN]   : Ban tu keo thanh ben PHAI len")
    print(" - Phim [MUI TEN XUONG] : Ban tu keo thanh ben PHAI xuong")
    print(" - BUONG TAY            : AI tu dong nhan dien bong va do giup ban")
    print(" - Phim [ESC]           : Thoat game")
    print("="*55 + "\n")

    while not done:
        # 1. Xử lý các sự kiện bàn phím
        action = 0  # mặc định đứng im (NOOP)
        human_intervened = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True

        # kiểm tra phím bấm liên tục của người chơi
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action = 2  # Hành động đi Lên của thanh Phải
            human_intervened = True
        elif keys[pygame.K_DOWN]:
            action = 3  # Hành động đi Xuống của thanh Phải
            human_intervened = True

        # 2. Lấy khung hình hiện tại để vẽ và cho AI phân tích
        frame_rgb = env.render()
        
        # Nếu người chơi không bấm phím, kích hoạt ngầm AI Heuristic siêu việt đỡ bóng
        if not human_intervened and frame_rgb is not None:
            # trích xuất vùng chơi bóng (bỏ phần viền trên dưới)
            play_area = frame_rgb[34:194, :, :]
            # tìm pixel của quả bóng (màu sáng đặc trưng)
            ball_pixels = np.where(play_area[:, :, 0] > 200)
            
            if len(ball_pixels[0]) > 0:
                ball_y = np.mean(ball_pixels[0]) + 34
                # định vị thanh trượt bên phải (cột X khoảng 140-145)
                paddle_pixels = np.where(frame_rgb[34:194, 140:145, 0] > 100)
                if len(paddle_pixels[0]) > 0:
                    paddle_y = np.mean(paddle_pixels[0]) + 34
                    # AI bám đuổi tọa độ bóng
                    if paddle_y > ball_y + 3:
                        action = 2  # UP
                    elif paddle_y < ball_y - 3:
                        action = 3  # DOWN

        # 3. Gửi bước đi vào game (chỉ tiến 1 frame duy nhất nên cực kỳ mượt và nhạy)
        _, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # cập nhật bảng điểm
        if reward > 0:
            right_score += 1
        elif reward < 0:
            left_score += 1

        # 4. Render hình ảnh ra Pygame
        if frame_rgb is not None:
            # xoay mảng pixel sang chuẩn Pygame (W, H, C)
            frame_surface = np.transpose(frame_rgb, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame_surface)
            surf = pygame.transform.scale(surf, (screen_width, screen_height))
            screen.blit(surf, (0, 0))

            # in trạng thái ai đang cầm lái
            controller_str = "BAN DIEU KHIEN" if human_intervened else "AI TU DONG"
            color = (0, 255, 0) if human_intervened else (0, 255, 255)
            
            # hiển thị dòng text trên ảnh
            mode_text = font.render(f"Che do: {controller_str}", True, color)
            score_text = font.render(f"May Atari: {left_score}  |  Ban & AI: {right_score}", True, (255, 255, 0))
            
            screen.blit(mode_text, (15, 10))
            screen.blit(score_text, (15, 35))

            pygame.display.flip()

        # Khóa FPS ở mức 30 hoặc 40 giúp game trôi qua với nhịp điệu hoàn hảo
        clock.tick(35)

    print(f"\nKET THUC DEMO! Ti so chung cuoc - May Atari: {left_score} | Ban & AI: {right_score}")
    env.close()
    pygame.quit()

if __name__ == "__main__":
    play()
