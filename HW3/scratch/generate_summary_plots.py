import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_summary_plots(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Soft Penalty Comparison
    plt.figure(figsize=(10, 6))
    colors = ['#0B7285', '#087F5B', '#5C940D', '#D9480F', '#5F3DC4']
    bars = plt.bar(df['seed'].astype(str), df['best_soft_penalty'], color=colors[:len(df)])
    plt.title('So sánh Soft Penalty giữa các Seed', fontsize=14, fontweight='bold')
    plt.xlabel('Seed', fontsize=12)
    plt.ylabel('Soft Penalty (Càng thấp càng tốt)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 4), ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'soft_penalty_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Convergence Speed (Generations run)
    plt.figure(figsize=(10, 6))
    plt.bar(df['seed'].astype(str), df['generations_run'], color='#364FC7', alpha=0.8)
    plt.title('Số thế hệ cần thiết để hội tụ', fontsize=14, fontweight='bold')
    plt.xlabel('Seed', fontsize=12)
    plt.ylabel('Số thế hệ', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_speed.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    csv_path = "outputs/report_ga_summary.csv"
    output_dir = "report/images"
    generate_summary_plots(csv_path, output_dir)
    print(f"Summary plots generated in {output_dir}")
