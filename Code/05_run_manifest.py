#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import cupy as cp

# 基于脚本位置计算路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_05_DIR = os.path.join(PROJECT_DIR, "data", "05")

# 添加ManiFeSt模块路径
sys.path.append(os.path.join(DATA_05_DIR, "ManiFeSt"))
from ManiFeSt_gpu import ManiFeSt_gpu

# ============ 配置 ============
CANCER_TYPE = "LUAD"
STATE_A = 1
STATE_B = 2
# ==============================

def load_data(file_path):
    """加载CSV数据并处理"""
    df = pd.read_csv(file_path, index_col=0)
    return df.values.astype(np.float32), df.columns.tolist()

def run_manifest(state_a, state_b, cancer_type=CANCER_TYPE):
    """对指定的两个state运行ManiFeSt特征选择"""
    print(f"\n{'='*60}")
    print(f"ManiFeSt: {cancer_type} State {state_a} vs State {state_b}")
    print(f"{'='*60}")

    feature_dir = os.path.join(DATA_05_DIR, 'feature_selection')
    file_a = os.path.join(feature_dir, f'{cancer_type}_state_{state_a}.csv')
    file_b = os.path.join(feature_dir, f'{cancer_type}_state_{state_b}.csv')

    X1, feature_names = load_data(file_a)
    X2, _ = load_data(file_b)

    X_combined = np.vstack([X1, X2])
    y_combined = np.hstack([np.zeros(X1.shape[0]), np.ones(X2.shape[0])])

    X_gpu = cp.asarray(X_combined, dtype=cp.float32)
    y_gpu = cp.asarray(y_combined, dtype=cp.int32)

    print(f"State {state_a}: {X1.shape[0]} samples, State {state_b}: {X2.shape[0]} samples")
    print(f"Features: {X_combined.shape[1]}")

    score = ManiFeSt_gpu(X_gpu, y_gpu)
    score_cpu = score.get() if hasattr(score, 'get') else score

    sorted_indices = np.argsort(score_cpu)[::-1]
    sorted_scores = score_cpu[sorted_indices]

    results = pd.DataFrame({
        'Rank': range(1, len(sorted_indices) + 1),
        'Feature_Index': sorted_indices,
        'Feature_Name': [feature_names[i] for i in sorted_indices],
        'ManiFeSt_Score': sorted_scores
    })

    output_file = os.path.join(DATA_05_DIR, f'{cancer_type}_manifest_state{state_a}_vs_state{state_b}.csv')
    results.to_csv(output_file, index=False)
    print(f"结果已保存: {output_file} ({len(results)} features)")
    print(f"Top 5: {', '.join(results['Feature_Name'].head(5).tolist())}")
    return results

def main():
    run_manifest(STATE_A, STATE_B)

if __name__ == "__main__":
    main() 
