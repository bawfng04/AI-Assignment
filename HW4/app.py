import streamlit as st
import os
import glob
from pathlib import Path

st.set_page_config(page_title="D3QN Atari Agent", layout="wide", page_icon="🎮")

st.title("🎮 Dueling Double DQN (D3QN) - Atari RL Agent")
st.markdown("**Trí tuệ Nhân tạo - Bài Tập Lớn 4**")

st.sidebar.header("Tùy chọn")
menu = st.sidebar.radio("Điều hướng", ["📊 Giám sát Huấn luyện", "🕹️ Đánh giá (Evaluation)"])

if menu == "📊 Giám sát Huấn luyện":
    st.header("Kết quả Huấn luyện (Training Metrics)")
    st.markdown("Biểu đồ trích xuất từ file log hiển thị sự cải thiện của mô hình theo thời gian.")
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("plot_reward.png"):
            st.image("plot_reward.png", caption="Tiến trình học (Episode Reward)", use_container_width=True)
        else:
            st.info("Chưa tìm thấy biểu đồ Reward. Hãy chạy `python plot_logs.py`.")
            
        if os.path.exists("plot_loss.png"):
            st.image("plot_loss.png", caption="Hàm mất mát (Huber Loss)", use_container_width=True)
    with col2:
        if os.path.exists("plot_epsilon.png"):
            st.image("plot_epsilon.png", caption="Giảm dần Epsilon (Exploration)", use_container_width=True)
            
        if os.path.exists("pipeline.png"):
            st.image("pipeline.png", caption="Kiến trúc D3QN Pipeline", use_container_width=True)

elif menu == "🕹️ Đánh giá (Evaluation)":
    st.header("Xem mô hình chơi game thực tế")
    st.markdown("Dùng script `eval.py` để ghi lại gameplay và xem trực tiếp tại đây.")
    
    st.code("python eval.py --checkpoint checkpoints/d3qn_ALE/Pong-v5_20000.pt --record gameplay.mp4", language="bash")
    
    videos = glob.glob("*.mp4") + glob.glob("*.gif") + glob.glob("assets/*.mp4") + glob.glob("assets/*.gif")
    if videos:
        st.success(f"Tìm thấy {len(videos)} bản ghi hình!")
        for vid in videos:
            st.subheader(f"Bản ghi: {vid}")
            if vid.endswith(".mp4"):
                st.video(vid)
            else:
                st.image(vid)
    else:
        st.warning("Chưa có video nào. Hãy chạy lệnh trên terminal để render video nhé!")
