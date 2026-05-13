"""
🎮 D3QN Atari RL Agent — Interactive Dashboard
Bài Tập Lớn 4: Trí tuệ Nhân tạo
"""

import streamlit as st
import os
import re
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────── PAGE CONFIG ───────────────────
st.set_page_config(
    page_title="D3QN Atari Agent",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────── CUSTOM CSS ───────────────────
st.markdown("""
<style>
    /* Light professional theme */
    .stApp {
        background: #f8f9fc;
    }
    
    /* Ép toàn bộ màu chữ về tone tối sang trọng (tránh lỗi chữ trắng của theme cũ) */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #1e293b !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }
    
    /* Ngoại trừ các nhãn đặc thù cần giữ màu nguyên bản hoặc gradient */
    .metric-card .value {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #64748b !important;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        color: #1e293b;
        border-left: 4px solid #4f46e5;
        padding-left: 14px;
        margin: 20px 0 12px 0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Comparison table */
    .comp-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        background: #ffffff;
        border: 1px solid #e2e8f0;
    }
    .comp-table th {
        background: #f1f5f9;
        color: #334155;
        padding: 12px 16px;
        text-align: center;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 2px solid #e2e8f0;
    }
    .comp-table td {
        padding: 10px 16px;
        text-align: center;
        color: #475569;
        border-bottom: 1px solid #f1f5f9;
    }
    .comp-table tr:hover td {
        background: #f8fafc;
    }
    .best-val { color: #059669 !important; font-weight: 700; }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: #94a3b8;
        font-size: 0.75rem;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)



# ─────────────────── LOG PARSER ───────────────────
def parse_log(filepath: str):
    """Parse training log file and return dict of metrics."""
    eps, rewards, avg100s, epsilons, losses = [], [], [], [], []
    if not os.path.exists(filepath):
        return None
    
    # Format: ep=1, reward=-21.0, avg100=-20.8, eps=0.997, loss=0.0001
    pat = re.compile(
        r"ep=(\d+),\s*reward=([\-\d\.]+)"
        r"(?:,\s*avg100=([\-\d\.]+))?"
        r"(?:,\s*eps=([\d\.]+))?"
        r"(?:,\s*loss=([\d\.]+))?"
    )
    
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                eps.append(int(m.group(1)))
                rewards.append(float(m.group(2)))
                avg100s.append(float(m.group(3)) if m.group(3) else None)
                epsilons.append(float(m.group(4)) if m.group(4) else None)
                losses.append(float(m.group(5)) if m.group(5) else None)
    
    if not eps:
        return None
    return {
        "episodes": eps,
        "rewards": rewards,
        "avg100": avg100s,
        "epsilon": epsilons,
        "loss": losses,
    }


def smooth(values, window=50):
    """Moving average smoothing."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid").tolist()


# ─────────────────── SIDEBAR ───────────────────
with st.sidebar:
    st.markdown("## 🎮 D3QN Dashboard")
    st.markdown("---")
    page = st.radio(
        "Điều hướng",
        [
            "🏠 Tổng quan Hệ thống",
            "📊 Training Analytics",
            "⚔️ So sánh 3 Models",
            "🕹️ Gameplay Demo",
        ],
        index=0,
    )
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#4a5568; font-size:0.75rem;'>"
        "Bài Tập Lớn 4<br>Trí tuệ Nhân tạo<br>Dueling Double DQN</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════
# PAGE 1: TỔNG QUAN HỆ THỐNG
# ═══════════════════════════════════════════════════
if page == "🏠 Tổng quan Hệ thống":
    st.markdown("# 🏗️ Tổng quan Kiến trúc D3QN")
    st.markdown(
        "Dueling Double Deep Q-Network với Prioritized Experience Replay "
        "trên môi trường **ALE/Pong-v5**."
    )

    # ── KPI METRICS ──
    d3qn_data = parse_log("logs/training.log") or parse_log("d3qn.log")
    best_reward = max(d3qn_data["rewards"]) if d3qn_data else "N/A"
    final_avg = d3qn_data["avg100"][-1] if d3qn_data and d3qn_data["avg100"][-1] else "N/A"
    total_eps = d3qn_data["episodes"][-1] if d3qn_data else "N/A"
    
    # Count checkpoints
    ckpt_count = len([f for f in os.listdir("checkpoints") if f.endswith(".pt")]) if os.path.isdir("checkpoints") else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="value">{best_reward}</div>'
            f'<div class="label">Best Reward</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="value">{final_avg}</div>'
            f'<div class="label">Final Avg-100</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="value">{total_eps}</div>'
            f'<div class="label">Total Episodes</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><div class="value">{ckpt_count}</div>'
            f'<div class="label">Checkpoints</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── ARCHITECTURE PIPELINE ──
    st.markdown('<div class="section-header">📐 Kiến trúc Pipeline</div>', unsafe_allow_html=True)
    if os.path.exists("pipeline.png"):
        st.image("pipeline.png", use_container_width=True)
    elif os.path.exists("diagram.png"):
        st.image("diagram.png", use_container_width=True)

    # ── KEY TECHNIQUES ──
    st.markdown('<div class="section-header">🧠 Kỹ thuật Cốt lõi</div>', unsafe_allow_html=True)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("#### 🔀 Double DQN")
        st.markdown(
            "Khử **Overestimation Bias** bằng cách tách biệt việc "
            "chọn hành động (Online Net) và đánh giá giá trị (Target Net)."
        )
        st.latex(r"Y = R_{t+1} + \gamma \, Q_{\text{target}}\!\left(S_{t+1},\; \arg\max_{a} Q_{\text{online}}(S_{t+1}, a)\right)")
    with col_b:
        st.markdown("#### 🏗️ Dueling Architecture")
        st.markdown(
            "Tách Q-Value thành 2 luồng: **Value V(s)** đánh giá trạng thái "
            "và **Advantage A(s,a)** đánh giá hành động."
        )
        st.latex(r"Q(s,a) = V(s) + \left(A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')\right)")
    with col_c:
        st.markdown("#### ⚡ Prioritized ER")
        st.markdown(
            "Lấy mẫu ưu tiên dựa trên **TD-Error** thông qua cấu trúc SumTree. "
            "Mẫu sai nhiều ➔ học nhiều hơn."
        )
        st.latex(r"P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta")


# ═══════════════════════════════════════════════════
# PAGE 2: TRAINING ANALYTICS
# ═══════════════════════════════════════════════════
elif page == "📊 Training Analytics":
    st.markdown("# 📊 Phân tích Chi tiết Quá trình Huấn luyện")

    d3qn_data = parse_log("logs/training.log") or parse_log("d3qn.log")
    if not d3qn_data:
        st.error("Không tìm thấy file log! Đảm bảo `logs/training.log` hoặc `d3qn.log` tồn tại.")
        st.stop()

    window = st.slider("🎚️ Cửa sổ Moving Average", 10, 200, 50, step=10)

    # ── REWARD & AVG100 ──
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Episode Reward", "Average Reward (100 eps)",
        "Epsilon Decay", "Training Loss"
    ), vertical_spacing=0.12)

    eps = d3qn_data["episodes"]
    
    # Reward
    fig.add_trace(go.Scatter(
        x=eps, y=d3qn_data["rewards"], mode="lines",
        line=dict(color="rgba(99,179,237,0.2)", width=1), name="Raw Reward", showlegend=False,
    ), row=1, col=1)
    sm_rew = smooth(d3qn_data["rewards"], window)
    fig.add_trace(go.Scatter(
        x=eps[window - 1:], y=sm_rew, mode="lines",
        line=dict(color="#63b3ed", width=2.5), name=f"Reward (MA-{window})",
    ), row=1, col=1)

    # Avg100
    avg_vals = [v for v in d3qn_data["avg100"] if v is not None]
    if avg_vals:
        fig.add_trace(go.Scatter(
            x=eps[: len(avg_vals)], y=avg_vals, mode="lines",
            line=dict(color="#68d391", width=2), name="Avg-100",
        ), row=1, col=2)

    # Epsilon
    eps_vals = [v for v in d3qn_data["epsilon"] if v is not None]
    if eps_vals:
        fig.add_trace(go.Scatter(
            x=eps[: len(eps_vals)], y=eps_vals, mode="lines",
            line=dict(color="#f6ad55", width=2), name="Epsilon",
        ), row=2, col=1)

    # Loss
    loss_vals = [v for v in d3qn_data["loss"] if v is not None]
    if loss_vals:
        fig.add_trace(go.Scatter(
            x=eps[: len(loss_vals)], y=loss_vals, mode="lines",
            line=dict(color="rgba(237,100,166,0.3)", width=1), name="Raw Loss", showlegend=False,
        ), row=2, col=2)
        sm_loss = smooth(loss_vals, min(window, len(loss_vals)))
        if len(sm_loss) > 0:
            fig.add_trace(go.Scatter(
                x=eps[min(window, len(loss_vals)) - 1: min(window, len(loss_vals)) - 1 + len(sm_loss)],
                y=sm_loss, mode="lines",
                line=dict(color="#ed64a6", width=2.5), name=f"Loss (MA-{window})",
            ), row=2, col=2)

    fig.update_layout(
        height=700,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fc",
        font=dict(color="#334155"),
        margin=dict(t=50, b=30),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5,
            font=dict(size=11),
        ),
    )
    st.plotly_chart(fig)

    # ── STATISTICS TABLE ──
    st.markdown('<div class="section-header">📋 Thống kê Tổng hợp</div>', unsafe_allow_html=True)
    rews = d3qn_data["rewards"]
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Best Reward", f"{max(rews):.1f}")
    sc2.metric("Worst Reward", f"{min(rews):.1f}")
    sc3.metric("Mean Reward", f"{np.mean(rews):.2f}")
    sc4.metric("Final Avg-100", f"{avg_vals[-1]:.2f}" if avg_vals else "N/A")


# ═══════════════════════════════════════════════════
# PAGE 3: SO SÁNH 3 MODELS
# ═══════════════════════════════════════════════════
elif page == "⚔️ So sánh 3 Models":
    st.markdown("# ⚔️ So sánh Hiệu năng: D3QN vs Baselines")

    log_configs = {
        "D3QN (Ours)": {"paths": ["logs/training.log", "d3qn.log"], "color": "#1f77b4"},
        "Double DQN": {"paths": ["logs/double_dqn/training.log"], "color": "#ff7f0e"},
        "Vanilla DQN": {"paths": ["logs/vanilla_dqn/training.log"], "color": "#2ca02c"},
    }

    window = st.slider("🎚️ Cửa sổ Moving Average", 10, 200, 50, step=10, key="comp_slider")

    fig = go.Figure()
    model_stats = {}

    for label, cfg in log_configs.items():
        data = None
        for p in cfg["paths"]:
            data = parse_log(p)
            if data:
                break
        if not data:
            continue

        rews = data["rewards"]
        eps_list = data["episodes"]
        color = cfg["color"]

        # Raw (transparent)
        fig.add_trace(go.Scatter(
            x=eps_list, y=rews, mode="lines",
            line=dict(color=color, width=1), opacity=0.15,
            name=f"{label} (raw)", showlegend=False,
        ))
        # Smoothed
        sm = smooth(rews, window)
        fig.add_trace(go.Scatter(
            x=eps_list[window - 1:], y=sm, mode="lines",
            line=dict(color=color, width=3), name=f"{label} (MA-{window})",
        ))

        model_stats[label] = {
            "best": max(rews),
            "final_100": np.mean(rews[-100:]),
            "mean": np.mean(rews),
            "episodes": len(rews),
        }

    fig.update_layout(
        height=500,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fc",
        font=dict(color="#334155"),
        xaxis_title="Episodes",
        yaxis_title="Episode Reward",
        title=dict(text="Performance Comparison on ALE/Pong-v5", font=dict(size=18)),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5,
            font=dict(size=12),
        ),
    )
    st.plotly_chart(fig)

    # ── COMPARISON TABLE ──
    if model_stats:
        st.markdown('<div class="section-header">📋 Bảng So sánh Chi tiết</div>', unsafe_allow_html=True)

        def best_class(vals, idx, maximize=True):
            target = max(vals) if maximize else min(vals)
            return ' class="best-val"' if vals[idx] == target else ""

        rows = list(model_stats.items())
        bests = [r[1]["best"] for r in rows]
        finals = [r[1]["final_100"] for r in rows]
        means = [r[1]["mean"] for r in rows]

        table_html = '<table class="comp-table"><tr><th>Model</th><th>Best Reward</th><th>Final Avg-100</th><th>Mean Reward</th><th>Episodes</th></tr>'
        for i, (name, s) in enumerate(rows):
            table_html += (
                f'<tr><td><b>{name}</b></td>'
                f'<td{best_class(bests, i)}>{s["best"]:.1f}</td>'
                f'<td{best_class(finals, i)}>{s["final_100"]:.2f}</td>'
                f'<td{best_class(means, i)}>{s["mean"]:.2f}</td>'
                f'<td>{s["episodes"]}</td></tr>'
            )
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

        # ── KEY INSIGHT ──
        st.markdown("")
        st.info(
            "💡 **Nhận xét:** D3QN vượt trội hẳn so với Vanilla DQN và Double DQN nhờ sự kết hợp "
            "3 kỹ thuật tiên tiến: Dueling Architecture (tách Value/Advantage), Double Target "
            "(khử Overestimation Bias), và Prioritized Experience Replay (học ưu tiên từ mẫu khó). "
            "Vanilla DQN bị kẹt ở mức reward thấp do thiếu cơ chế khử thiên lệch."
        )

    # ── STATIC COMPARISON IMAGE ──
    if os.path.exists("comparison_curve.png"):
        with st.expander("📈 Xem biểu đồ tĩnh (Matplotlib)", expanded=False):
            st.image("comparison_curve.png", use_container_width=True)


# ═══════════════════════════════════════════════════
# PAGE 4: GAMEPLAY DEMO
# ═══════════════════════════════════════════════════
elif page == "🕹️ Gameplay Demo":
    st.markdown("# 🕹️ D3QN Agent chơi Atari Pong")
    st.markdown(
        "Video dưới đây được ghi lại bởi agent D3QN đã huấn luyện xong, "
        "sử dụng chính sách **greedy** (ε = 0) với checkpoint cuối cùng."
    )

    video_files = [f for f in ["video.mp4", "gameplay.mp4", "assets/gameplay.mp4"] if os.path.exists(f)]
    
    if video_files:
        import base64
        for vf in video_files:
            with open(vf, "rb") as f:
                video_bytes = f.read()
            b64_video = base64.b64encode(video_bytes).decode()
            
            # Khóa chặt chiều cao tối đa (max-height: 400px), tự động căn giữa mượt mà
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; margin: 20px 0;">
                    <video controls autoplay loop muted style="max-height: 400px; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.12); border: 1px solid #e2e8f0;">
                        <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
                        Trình duyệt của bạn không hỗ trợ thẻ video.
                    </video>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"📹 Nguồn video nhúng: `{vf}`")
    else:
        st.warning(
            "⚠️ Chưa có video gameplay. Hãy chạy evaluation trên Colab:\n"
            "```python\n"
            "rewards = evaluate(agent, config['env']['name'], n_episodes=5, seed=42, record_path='gameplay.mp4')\n"
            "```"
        )

    # ── CHECKPOINT INFO ──
    st.markdown('<div class="section-header">💾 Danh sách Checkpoint khả dụng</div>', unsafe_allow_html=True)
    if os.path.isdir("checkpoints"):
        ckpts = sorted(
            [f for f in os.listdir("checkpoints") if f.endswith(".pt")],
            key=lambda x: os.path.getmtime(os.path.join("checkpoints", x)),
        )
        if ckpts:
            for ck in ckpts:
                size_mb = os.path.getsize(os.path.join("checkpoints", ck)) / (1024 * 1024)
                st.markdown(f"- `{ck}` — **{size_mb:.1f} MB**")
        else:
            st.info("Chưa có checkpoint nào.")
    else:
        st.info("Thư mục `checkpoints/` không tồn tại.")


# ─────────────────── FOOTER ───────────────────
st.markdown(
    '<div class="footer-text">'
    "🎮 Dueling Double DQN (D3QN) with PER — Atari Pong Agent<br>"
    "Bài Tập Lớn 4 · Trí tuệ Nhân tạo"
    "</div>",
    unsafe_allow_html=True,
)
