import torch
import time
import subprocess
import os
from datetime import datetime

def clear_screen():
    """清空终端屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_gpu_usage():
    """获取GPU使用情况"""
    try:
        result = subprocess.run(['nvidia-smi'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True)
        return result.stdout
    except FileNotFoundError:
        return "nvidia-smi not found"

def keep_gpu_busy(interval=0.05, display_interval=5):
    """
    持续占用GPU资源并显示使用情况
    
    参数:
        interval: 每次计算间隔时间(秒)
        display_interval: 显示GPU状态的间隔时间(秒)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    last_display = 0

    a = torch.randn(8192, 8192, device=device)
    b = torch.randn(8192, 8192, device=device)

    try:
        while True:
            current_time = time.time()
            
            # 执行GPU计算
            c = torch.matmul(a, b)
            torch.cuda.synchronize(device)
            
            # 定期显示GPU状态
            if current_time - last_display >= display_interval:
                clear_screen()
                print(f"=== GPU Keep-Alive Monitor ===")
                print(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Calculation interval: {interval}s")
                print("\nCurrent GPU Usage:")
                print(get_gpu_usage())
                print("\nPress Ctrl+C to stop...")
                last_display = current_time
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nStopping GPU keep-alive...")

if __name__ == "__main__":
    keep_gpu_busy()
