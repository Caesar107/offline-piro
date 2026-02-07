"""
PIRO 训练曲线绘制脚本
用于可视化 PIRO 算法的训练效果
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_curve(csv_path, save_dir=None):
    """
    绘制训练曲线
    
    Args:
        csv_path: CSV 文件路径
        save_dir: 保存图片的目录，默认为 CSV 文件所在目录
    """
    # 读取数据
    df = pd.read_csv(csv_path)
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(df) + 1)
    
    # 1. True Reward 曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, df['Reward'], label='True Reward', color='#2196F3', alpha=0.8, linewidth=1)
    # 添加平滑曲线
    window = min(50, len(df) // 10)
    if window > 1:
        smoothed = df['Reward'].rolling(window=window, center=True).mean()
        ax1.plot(epochs, smoothed, label=f'Smoothed (window={window})', color='#1565C0', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reward')
    ax1.set_title('True Reward (Real Environment)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. WM Reward 曲线
    ax2 = axes[0, 1]
    ax2.plot(epochs, df['Reward_WM'], label='WM Reward', color='#FF9800', alpha=0.8, linewidth=1)
    if window > 1:
        smoothed_wm = df['Reward_WM'].rolling(window=window, center=True).mean()
        ax2.plot(epochs, smoothed_wm, label=f'Smoothed (window={window})', color='#E65100', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Reward')
    ax2.set_title('World Model Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 两者对比
    ax3 = axes[1, 0]
    ax3.plot(epochs, df['Reward'], label='True Reward', color='#2196F3', alpha=0.6, linewidth=1)
    ax3.plot(epochs, df['Reward_WM'], label='WM Reward', color='#FF9800', alpha=0.6, linewidth=1)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Reward')
    ax3.set_title('True Reward vs WM Reward')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. L2 Coefficient 变化
    ax4 = axes[1, 1]
    if 'L2_Coef' in df.columns:
        ax4.plot(epochs, df['L2_Coef'], label='L2 Coefficient', color='#4CAF50', alpha=0.8, linewidth=1.5)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Coefficient')
        ax4.set_title('L2 Coefficient (Trust Region)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No L2_Coef data', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, 'piro_training_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n图片已保存到: {save_path}")
    
    # 打印统计信息
    print("\n" + "="*50)
    print("训练统计")
    print("="*50)
    print(f"总 Epoch 数: {len(df)}")
    print(f"\nTrue Reward:")
    print(f"  最大值: {df['Reward'].max():.2f}")
    print(f"  最小值: {df['Reward'].min():.2f}")
    print(f"  平均值: {df['Reward'].mean():.2f}")
    print(f"  最后100个epoch平均: {df['Reward'].tail(100).mean():.2f}")
    print(f"  最后50个epoch平均: {df['Reward'].tail(50).mean():.2f}")
    
    print(f"\nWM Reward:")
    print(f"  最大值: {df['Reward_WM'].max():.2f}")
    print(f"  最小值: {df['Reward_WM'].min():.2f}")
    print(f"  平均值: {df['Reward_WM'].mean():.2f}")
    
    # 显示图片（如果在交互环境中）
    try:
        plt.show()
    except:
        pass
    
    return df


def plot_comparison(piro_csv, mlirl_csv, save_dir=None):
    """
    对比 PIRO 和 ML-IRL 的训练曲线
    
    Args:
        piro_csv: PIRO 的 CSV 文件路径
        mlirl_csv: ML-IRL 的 CSV 文件路径
        save_dir: 保存目录
    """
    df_piro = pd.read_csv(piro_csv)
    df_mlirl = pd.read_csv(mlirl_csv)
    
    if save_dir is None:
        save_dir = os.path.dirname(piro_csv)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. True Reward 对比
    ax1 = axes[0]
    epochs_piro = range(1, len(df_piro) + 1)
    epochs_mlirl = range(1, len(df_mlirl) + 1)
    
    ax1.plot(epochs_piro, df_piro['Reward'], label='PIRO', color='#2196F3', alpha=0.6, linewidth=1)
    ax1.plot(epochs_mlirl, df_mlirl['Reward'], label='ML-IRL', color='#F44336', alpha=0.6, linewidth=1)
    
    # 添加平滑曲线
    window = 50
    if len(df_piro) > window:
        smoothed_piro = df_piro['Reward'].rolling(window=window, center=True).mean()
        ax1.plot(epochs_piro, smoothed_piro, label='PIRO (smoothed)', color='#1565C0', linewidth=2)
    if len(df_mlirl) > window:
        smoothed_mlirl = df_mlirl['Reward'].rolling(window=window, center=True).mean()
        ax1.plot(epochs_mlirl, smoothed_mlirl, label='ML-IRL (smoothed)', color='#B71C1C', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('True Reward')
    ax1.set_title('PIRO vs ML-IRL: True Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. WM Reward 对比
    ax2 = axes[1]
    ax2.plot(epochs_piro, df_piro['Reward_WM'], label='PIRO', color='#2196F3', alpha=0.6, linewidth=1)
    ax2.plot(epochs_mlirl, df_mlirl['Reward_WM'], label='ML-IRL', color='#F44336', alpha=0.6, linewidth=1)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('WM Reward')
    ax2.set_title('PIRO vs ML-IRL: World Model Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'piro_vs_mlirl_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n对比图已保存到: {save_path}")
    
    # 打印对比统计
    print("\n" + "="*50)
    print("PIRO vs ML-IRL 对比统计")
    print("="*50)
    print(f"\n{'指标':<25} {'PIRO':>15} {'ML-IRL':>15}")
    print("-"*55)
    print(f"{'最终 Reward (后100 epoch)':<25} {df_piro['Reward'].tail(100).mean():>15.2f} {df_mlirl['Reward'].tail(100).mean():>15.2f}")
    print(f"{'最大 Reward':<25} {df_piro['Reward'].max():>15.2f} {df_mlirl['Reward'].max():>15.2f}")
    print(f"{'平均 Reward':<25} {df_piro['Reward'].mean():>15.2f} {df_mlirl['Reward'].mean():>15.2f}")
    
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    # 默认路径
    default_csv = "E:\code\offline-piro\data\hopper_medexp42\hopper-medium-expert-v2_42_piro_offline.csv"
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = default_csv
    
    if not os.path.exists(csv_path):
        print(f"错误: 文件不存在 - {csv_path}")
        print(f"\n用法: python {sys.argv[0]} <csv_file_path>")
        print(f"示例: python {sys.argv[0]} /path/to/training_results.csv")
        sys.exit(1)
    
    print(f"正在处理: {csv_path}")
    plot_training_curve(csv_path)



