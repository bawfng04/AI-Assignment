import re
import matplotlib.pyplot as plt

def parse_logs(log_file):
    episodes = {}
    
    # Regex để tìm các thông số từ dòng log của tqdm
    # Ví dụ: ... ep=108, reward=-21.0, avg100=-20.7, eps=0.776, loss=0.0000]
    pattern = re.compile(r'ep=(\d+),\s*reward=([\-\.\d]+),\s*avg100=([\-\.\d]+),\s*eps=([\.\d]+),\s*loss=([\.\d]+)')
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ep = int(match.group(1))
                reward = float(match.group(2))
                avg100 = float(match.group(3))
                eps = float(match.group(4))
                loss = float(match.group(5))
                
                # Cập nhật dict sẽ tự động giữ lại giá trị cuối cùng của mỗi episode
                episodes[ep] = {
                    'reward': reward,
                    'avg100': avg100,
                    'eps': eps,
                    'loss': loss
                }
                
    return episodes

def plot_metrics(episodes):
    # Trích xuất dữ liệu thành mảng
    eps_list = sorted(episodes.keys())
    rewards = [episodes[e]['reward'] for e in eps_list]
    avg100 = [episodes[e]['avg100'] for e in eps_list]
    epsilons = [episodes[e]['eps'] for e in eps_list]
    losses = [episodes[e]['loss'] for e in eps_list]
    
    # 1. Biểu đồ Reward
    plt.figure(figsize=(10, 5))
    plt.plot(eps_list, rewards, alpha=0.3, color='blue', label='Episode Reward (Raw)')
    plt.plot(eps_list, avg100, color='red', linewidth=2, label='Avg Reward (Last 100)')
    plt.title('Episode Reward vs Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plot_reward.png', dpi=300)
    plt.close()
    print("Đã lưu biểu đồ: plot_reward.png")
    
    # 2. Biểu đồ Epsilon (Exploration Rate)
    plt.figure(figsize=(10, 5))
    plt.plot(eps_list, epsilons, color='green', linewidth=2)
    plt.title('Epsilon Decay vs Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plot_epsilon.png', dpi=300)
    plt.close()
    print("Đã lưu biểu đồ: plot_epsilon.png")
    
    # 3. Biểu đồ Loss
    plt.figure(figsize=(10, 5))
    plt.plot(eps_list, losses, color='purple', alpha=0.6)
    plt.title('Huber Loss vs Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plot_loss.png', dpi=300)
    plt.close()
    print("Đã lưu biểu đồ: plot_loss.png")

if __name__ == '__main__':
    log_file = 'heuristic.log'
    print(f"Đang phân tích file {log_file}...")
    
    ep_data = parse_logs(log_file)
    
    if ep_data:
        print(f"-> Đã tìm thấy dữ liệu của {len(ep_data)} episodes.")
        plot_metrics(ep_data)
    else:
        print("-> Lỗi: Không tìm thấy định dạng log hợp lệ trong file.")
