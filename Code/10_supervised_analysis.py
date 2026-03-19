#!/usr/bin/env python3
"""
监督降维分析脚本
使用监督学习方法来改善State聚类效果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.patches as mpatches

# 基于脚本位置计算路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_10_DIR = os.path.join(PROJECT_DIR, "data", "10")

def load_state_data(state_dir):
    """读取所有state文件并合并数据"""
    all_data = []
    state_labels = []
    
    # 获取所有state文件
    state_files = []
    for file in os.listdir(state_dir):
        if file.startswith("LUAD_state_") and file.endswith(".csv"):
            state_files.append(file)
    
    state_files.sort()  # 按文件名排序
    print(f"找到 {len(state_files)} 个state文件:")
    
    for state_file in state_files:
        # 提取state编号
        state_num = state_file.split("_state_")[1].split(".csv")[0]
        state_path = os.path.join(state_dir, state_file)
        
        print(f"  读取 {state_file} (State {state_num})")
        
        # 读取数据
        try:
            state_data = pd.read_csv(state_path)
            print(f"    样本数: {len(state_data)}, 特征数: {len(state_data.columns)-1}")
            
            # 获取样本ID和特征数据
            features = state_data.iloc[:, 1:].values   # 其余列是特征
            
            # 添加到总数据中
            all_data.append(features)
            state_labels.extend([int(state_num)] * len(features))  # 使用数字标签
            
        except Exception as e:
            print(f"    错误：读取文件 {state_file} 失败: {e}")
            continue
    
    # 合并所有数据
    if all_data:
        combined_data = np.vstack(all_data)
        print(f"\n合并后的数据维度: {combined_data.shape}")
        print(f"样本总数: {len(state_labels)}")
        return combined_data, np.array(state_labels)
    else:
        print("错误：没有成功读取任何数据")
        return None, None

def perform_lda(data, labels):
    """执行线性判别分析(LDA)"""
    print(f"\n执行线性判别分析(LDA)...")
    print(f"输入数据维度: {data.shape}")
    print(f"类别数: {len(np.unique(labels))}")
    
    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # LDA降维 (最多n_classes-1维)
    n_components = min(len(np.unique(labels)) - 1, 2)  # 对于可视化，限制为2维
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    
    try:
        lda_result = lda.fit_transform(data_scaled, labels)
        print(f"LDA完成，输出维度: {lda_result.shape}")
        print(f"解释方差比: {lda.explained_variance_ratio_}")
        return lda_result
    except Exception as e:
        print(f"LDA失败: {e}")
        return None

def save_custom_lda_plot(lda_result, labels, output_dir):
    """保存满足用户样式要求的单独LDA结果图。
    - 字体：Arial
    - 颜色：与提供示例一致（手工指定近似色）
    - 不显示坐标轴说明文字
    """
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['axes.linewidth'] = 3.8

    # 颜色映射（近似示例图颜色）
    state_colors = {
        1: '#FF6F6F',  # 1 近似珊瑚红
        2: '#B58900',  # 2 近似赭黄
        3: '#73C476',  # 3 近似草绿
        4: '#66C2A4',  # 4 近似青绿
        5: '#56B4E9',  # 5 天蓝
        6: '#8C6BB1',  # 6 紫色
        7: '#F17CB0',  # 7 品红/粉
    }

    unique_states = sorted(set(labels))
    palette = {s: state_colors.get(int(s), '#000000') for s in unique_states}

    plt.figure(figsize=(9, 9))
    for s in unique_states:
        mask = labels == s
        plt.scatter(
            lda_result[mask, 0],
            lda_result[mask, 1] if lda_result.shape[1] > 1 else 0,
            s=160,
            alpha=0.7,
            c=[palette[s]],
            label=str(s)
        )

    # 显示坐标轴
    ax = plt.gca()
    ax.set_xlabel('LD1', fontsize=18, fontweight='bold')
    ax.set_ylabel('LD2', fontsize=18, fontweight='bold')
    # 保留刻度和网格
    ax.grid(True, alpha=0.25)
    # 加粗并变黑四周边框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3.8)
    # 刻度风格（显示刻度与刻度标签）
    ax.tick_params(axis='both', which='major', labelsize=14, width=3.8, length=6)

    # 不在主图显示图例
    plt.tight_layout()

    out_file = os.path.join(output_dir, 'lda_states_arial.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"自定义LDA结果图已保存: {out_file}")

    # 额外：生成一个横向的State图例图片（使用方块）
    fig_legend = plt.figure(figsize=(12, 1.8))
    ax_leg = fig_legend.add_subplot(111)
    ax_leg.axis('off')
    # 手动在同一行绘制 "State" + 方块 + 数字
    ax_leg.set_xlim(0, 120)
    ax_leg.set_ylim(0, 1)
    # 标题文字
    ax_leg.text(5, 0.5, 'State', fontsize=20, va='center', ha='left')
    # 从一定偏移开始依次绘制方块与数字
    x = 25
    step = 13
    rect_w = 8
    rect_h = 0.35
    y = 0.5 - rect_h/2
    for s in unique_states:
        ax_leg.add_patch(mpatches.Rectangle((x, y), rect_w, rect_h, color=palette[s], ec='none'))
        ax_leg.text(x + rect_w + 2, 0.5, str(s), fontsize=18, va='center', ha='left')
        x += step
    legend_out = os.path.join(output_dir, 'lda_states_legend.png')
    fig_legend.savefig(legend_out, dpi=300, bbox_inches='tight')
    plt.close(fig_legend)
    print(f"横向State图例已保存: {legend_out}")

def perform_supervised_umap(data, labels):
    """执行监督UMAP"""
    print(f"\n执行监督UMAP...")
    print(f"输入数据维度: {data.shape}")
    
    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 监督UMAP
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42,
        verbose=True
    )
    
    # 使用标签进行监督学习
    supervised_umap_result = reducer.fit_transform(data_scaled, y=labels)
    print(f"监督UMAP完成，输出维度: {supervised_umap_result.shape}")
    
    return supervised_umap_result

def perform_tsne_with_different_perplexity(data, labels):
    """使用不同perplexity值的t-SNE"""
    print(f"\n尝试不同perplexity的t-SNE...")
    
    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    perplexity_values = [5, 10, 30, 50, 100]
    results = {}
    
    for perp in perplexity_values:
        if perp < len(data_scaled):
            print(f"  尝试perplexity={perp}...")
            try:
                tsne = TSNE(
                    n_components=2,
                    perplexity=perp,
                    n_iter=1000,
                    random_state=42,
                    verbose=0
                )
                tsne_result = tsne.fit_transform(data_scaled)
                results[f"perplexity_{perp}"] = tsne_result
                print(f"    完成")
            except Exception as e:
                print(f"    失败: {e}")
                continue
    
    return results

def create_method_comparison(methods_results, labels, output_dir):
    """创建不同方法的比较可视化"""
    print(f"\n创建方法比较可视化...")
    
    n_methods = len(methods_results)
    if n_methods == 0:
        print("没有结果可以可视化")
        return
    
    # 计算子图布局
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 准备颜色
    unique_states = sorted(set(labels))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_states)))
    color_map = dict(zip(unique_states, colors))
    
    for idx, (method_name, result) in enumerate(methods_results.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # 绘制每个state
        for state in unique_states:
            state_mask = labels == state
            ax.scatter(
                result[state_mask, 0],
                result[state_mask, 1],
                c=[color_map[state]],
                label=f"State_{state}",
                alpha=0.7,
                s=30
            )
        
        ax.set_title(method_name, fontsize=12)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.grid(True, alpha=0.3)
        
        # 只在第一个子图显示图例
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 隐藏多余的子图
    for idx in range(n_methods, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    
    # 保存比较图
    comparison_file = os.path.join(output_dir, "method_comparison.png")
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    print(f"方法比较图已保存: {comparison_file}")
    
    plt.close()

def calculate_silhouette_scores(methods_results, labels):
    """计算轮廓系数来评估聚类质量"""
    from sklearn.metrics import silhouette_score
    
    print(f"\n计算聚类质量评估...")
    scores = {}
    
    for method_name, result in methods_results.items():
        try:
            score = silhouette_score(result, labels)
            scores[method_name] = score
            print(f"{method_name}: 轮廓系数 = {score:.3f}")
        except Exception as e:
            print(f"{method_name}: 计算轮廓系数失败 - {e}")
            scores[method_name] = None
    
    return scores

def main():
    print("=" * 60)
    print("监督降维分析 - LUAD States")
    print("=" * 60)
    
    # 设置参数
    state_dir = os.path.join(DATA_10_DIR, "state_result")
    output_dir = os.path.join(DATA_10_DIR, "supervised_results")
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 检查输入目录
    if not os.path.exists(state_dir):
        print(f"错误：目录 {state_dir} 不存在")
        return
    
    # 读取数据
    print("1. 读取state数据...")
    data, labels = load_state_data(state_dir)
    
    if data is None:
        print("数据读取失败，程序退出")
        return
    
    print(f"\n数据统计:")
    print(f"- 数据维度: {data.shape}")
    print(f"- 类别分布:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  State {label}: {count} 样本")
    
    # 收集所有方法的结果
    methods_results = {}
    
    # 方法1: 线性判别分析 (LDA)
    print("\n" + "="*50)
    print("方法1: 线性判别分析 (LDA)")
    print("="*50)
    
    lda_result = perform_lda(data, labels)
    if lda_result is not None:
        methods_results["LDA"] = lda_result
        # 保存用户指定样式的单独LDA结果图
        try:
            save_custom_lda_plot(lda_result, labels, output_dir)
        except Exception as e:
            print(f"保存自定义LDA图失败: {e}")
    
    # 方法2: 监督UMAP
    print("\n" + "="*50)
    print("方法2: 监督UMAP")
    print("="*50)
    
    try:
        supervised_umap_result = perform_supervised_umap(data, labels)
        if supervised_umap_result is not None:
            methods_results["Supervised UMAP"] = supervised_umap_result
    except Exception as e:
        print(f"监督UMAP失败: {e}")
    
    # 方法3: 不同perplexity的t-SNE
    print("\n" + "="*50)
    print("方法3: 不同perplexity的t-SNE")
    print("="*50)
    
    tsne_results = perform_tsne_with_different_perplexity(data, labels)
    for name, result in tsne_results.items():
        methods_results[f"t-SNE ({name})"] = result
    
    # 创建比较可视化
    if methods_results:
        print("\n" + "="*50)
        print("创建比较可视化")
        print("="*50)
        
        create_method_comparison(methods_results, labels, output_dir)
        
        # 计算聚类质量评估
        scores = calculate_silhouette_scores(methods_results, labels)
        
        # 保存评估结果
        scores_file = os.path.join(output_dir, "clustering_scores.txt")
        with open(scores_file, 'w') as f:
            f.write("聚类质量评估 (轮廓系数)\n")
            f.write("=" * 30 + "\n\n")
            for method, score in scores.items():
                if score is not None:
                    f.write(f"{method}: {score:.3f}\n")
                else:
                    f.write(f"{method}: 计算失败\n")
            
            f.write("\n注释:\n")
            f.write("轮廓系数范围: [-1, 1]\n")
            f.write("值越高表示聚类效果越好\n")
            f.write("> 0.7: 强聚类结构\n")
            f.write("0.5-0.7: 合理聚类结构\n")
            f.write("0.25-0.5: 弱聚类结构\n")
            f.write("< 0.25: 无明显聚类结构\n")
        
        print(f"聚类评估结果已保存: {scores_file}")
    
    print("\n" + "="*60)
    print("监督降维分析完成！")
    print("="*60)
    print(f"输出目录: {output_dir}")
    print("生成的文件:")
    print("- method_comparison.png: 不同方法比较")
    print("- clustering_scores.txt: 聚类质量评估")
    
    if methods_results:
        print("\n最佳方法推荐:")
        valid_scores = {k: v for k, v in scores.items() if v is not None}
        if valid_scores:
            best_method = max(valid_scores.items(), key=lambda x: x[1])
            print(f"根据轮廓系数，最佳方法是: {best_method[0]} (分数: {best_method[1]:.3f})")
        else:
            print("无法确定最佳方法")

if __name__ == "__main__":
    main()