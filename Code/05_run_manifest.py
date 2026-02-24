#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import cupy as cp

# 添加ELVES模块路径（相对路径）
sys.path.append('../Data/ELVES/ManiFeSt')
from ManiFeSt_gpu import ManiFeSt_gpu

def load_data(file_path):
    """加载CSV数据并处理"""
    df = pd.read_csv(file_path, index_col=0)
    return df.values.astype(np.float32), df.columns.tolist()

def main():
    print("运行ManiFeSt特征选择算法...")
    
    # 加载数据（相对路径，相对于Code目录）
    X1, feature_names = load_data('../Data/feature_selection/LUAD_1.csv')
    X2, _ = load_data('../Data/feature_selection/LUAD_2.csv')
    
    # 合并数据并创建标签
    # X1 标签为 0, X2 标签为 1
    X_combined = np.vstack([X1, X2])  # 合并特征矩阵
    y_combined = np.hstack([np.zeros(X1.shape[0]), np.ones(X2.shape[0])])  # 创建标签
    
    # 转换为CuPy数组
    X_gpu = cp.asarray(X_combined, dtype=cp.float32)
    y_gpu = cp.asarray(y_combined, dtype=cp.int32)
    
    print(f"Combined data shape: {X_gpu.shape}")
    print(f"Labels shape: {y_gpu.shape}")
    print(f"Class 0 samples: {cp.sum(y_gpu == 0).get()}")
    print(f"Class 1 samples: {cp.sum(y_gpu == 1).get()}")
    
    # 运行ManiFeSt算法
    score = ManiFeSt_gpu(X_gpu, y_gpu)
    
    # 将结果从GPU转移到CPU
    if hasattr(score, 'get'):
        score_cpu = score.get()
    else:
        score_cpu = score
    
    # 按降序排序
    sorted_indices = np.argsort(score_cpu)[::-1]
    sorted_scores = score_cpu[sorted_indices]
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'Rank': range(1, len(sorted_indices) + 1),
        'Feature_Index': sorted_indices,
        'Feature_Name': [feature_names[i] for i in sorted_indices],
        'ManiFeSt_Score': sorted_scores
    })
    
    # 保存结果
    results.to_csv('../Data/output/manifest_results.csv', index=False)
    print(f"结果已保存到 ../Data/output/manifest_results.csv ({len(results)} 个特征)")

if __name__ == "__main__":
    main() 
