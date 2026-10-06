import json
import matplotlib.pyplot as plt
import numpy as np
import os

model_name = "Qwen3-14B"

# 为每个数据集创建折线图
output_dir = f"exp/plot/dataset_plots/{model_name}_async"
os.makedirs(output_dir, exist_ok=True)

# 定义要读取的JSON文件路径列表
json_file_paths = [
    # f"exp/results_longbench/{model_name}/pred_512/result.json",
    # f"exp/results_longbench/{model_name}/pred_1024/result.json",
    # f"exp/results_longbench/{model_name}/pred_2048/result.json",
    f"exp/results_longbench/{model_name}/pred_tls/result.json",
    # # Qwen3-32B
    # "exp/results_longbench/Qwen3-32B/pred_1024/result.json",
    # Qwen3-14B
    # "exp/results_longbench/Qwen3-14B/pred_512/result.json",
    # "exp/results_longbench/Qwen3-14B/pred_1024/result.json",
    # "exp/results_longbench/Qwen3-14B/pred_2048/result.json",
    # # Qwen3-8B
    # "exp/results_longbench/Qwen3-8B/pred_512/result.json",
    # "exp/results_longbench/Qwen3-8B/pred_1024/result.json",
    # "exp/results_longbench/Qwen3-8B/pred_2048/result.json",
]

# 解析数据，按数据集分组
dataset_data = {}

# 遍历所有JSON文件
for json_file_path in json_file_paths:
    print(f"正在读取文件: {json_file_path}")
    
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for key, value in data.items():
            # 解析key：数据集名-方法名-时间戳.jsonl
            # 首先去掉.jsonl后缀
            key_without_ext = key.replace('.jsonl', '')
            parts = key_without_ext.split('-')

            # 通用处理：合并前两部分作为数据集名
            dataset_name = parts[0]
            method_name = parts[1].split("_")[0]
            
            if method_name == "tia":
                if parts[1].endswith("_c4"):
                    method_name = f"{method_name}_c4"
                elif parts[1].endswith("_c2"):
                    method_name = f"{method_name}_c2"
                elif parts[1].endswith("_c4_async"):
                    method_name = f"{method_name}_c4_async"
                elif parts[1].endswith("_c2_async"):
                    method_name = f"{method_name}_c2_async"
                if method_name == "tia":
                    method_name = f"{method_name}_c2"

            if dataset_name not in dataset_data:
                dataset_data[dataset_name] = {}
            
            if method_name not in dataset_data[dataset_name]:
                dataset_data[dataset_name][method_name] = []
            
            dataset_data[dataset_name][method_name].append({
                'budget': value['budget'],
                'score': value['score'],
                'source_file': json_file_path  # 记录数据来源
            })
            
    except FileNotFoundError:
        print(f"警告: 文件不存在 - {json_file_path}")
    except json.JSONDecodeError:
        print(f"错误: JSON解析失败 - {json_file_path}")
    except Exception as e:
        print(f"错误: 处理文件时出错 - {json_file_path}: {str(e)}")

print(f"\n成功读取 {len(json_file_paths)} 个文件")
print(f"发现 {len(dataset_data)} 个数据集")

# 定义颜色和标记样式
colors = ['#4a7dba', '#8fc7de', '#ffd680', '#fa874f', '#d93026']
markers = ['o']

for dataset_name, methods in dataset_data.items():
    plt.figure(figsize=(10, 6))
    
    # 分离baseline方法和普通方法
    baseline_methods = {}
    regular_methods = {}
    
    for method_name, points in methods.items():
        # 检查是否有budget为-1的点（baseline）
        has_baseline = any(p['budget'] == -1 for p in points)
        if has_baseline:
            baseline_methods[method_name] = points
        else:
            regular_methods[method_name] = points
    
    # 首先绘制普通方法的折线图
    for idx, (method_name, points) in enumerate(regular_methods.items()):
        # 按budget排序
        sorted_points = sorted(points, key=lambda x: x['budget'])
        budgets = [p['budget'] for p in sorted_points]
        scores = [p['score'] for p in sorted_points]
        
        # 绘制折线图
        plt.plot(budgets, scores, 
                marker=markers[idx % len(markers)], 
                color=colors[idx % len(colors)],
                linewidth=2,
                markersize=8,
                label=method_name)
    
    # 然后绘制baseline方法的横线
    for idx, (method_name, points) in enumerate(baseline_methods.items()):
        # 找到baseline的score（budget为-1的点）
        baseline_point = next(p for p in points if p['budget'] == -1)
        baseline_score = baseline_point['score']
        
        # 获取当前x轴范围
        # x_min, x_max = plt.xlim()
        x_min = 400
        x_max = 2200
        
        # 绘制水平横线
        plt.hlines(y=baseline_score, 
                  xmin=x_min, 
                  xmax=x_max, 
                  colors=colors[(idx + len(regular_methods)) % len(colors)],
                  linestyles='--',
                  linewidth=2,
                  label=f"{method_name} (baseline)")
    
    plt.xlabel('Budget', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Dataset: {dataset_name} - Score vs Budget', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Methods', fontsize=10, title_fontsize=11)
    
    # 设置x轴范围，排除负值
    all_budgets = []
    for points in methods.values():
        all_budgets.extend([p['budget'] for p in points])
    
    # 过滤掉负值（包括-1）
    positive_budgets = [b for b in all_budgets if b > 0]
    if positive_budgets:
        min_budget = min(positive_budgets)
        max_budget = max(positive_budgets)
        
        # 添加一些边距
        plt.xlim(400, 2200)
    
    # 自动调整y轴范围
    all_scores = [p['score'] for points in methods.values() for p in points]
    if all_scores:
        plt.ylim(min(all_scores) * 0.9, max(all_scores) * 1.1)
    
    # 保存图像
    filename = f"{dataset_name}_score_vs_budget.png"
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存: {filepath}")

print(f"\n所有图表已保存到 '{output_dir}' 目录中")
print(f"共生成 {len(dataset_data)} 个数据集的折线图")