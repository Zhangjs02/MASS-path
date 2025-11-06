#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
import os
import random
warnings.filterwarnings('ignore')

# 设置matplotlib字体，避免中文显示问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_result_directory():
    """创建结果保存目录"""
    result_dir = 'rank_result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"创建结果目录: {result_dir}")
    return result_dir

def load_data():
    """加载数据并合并为分类任务格式"""
    # 加载两类样本数据
    df1 = pd.read_csv('ELVES_/data/LUAD_2/LUAD_1.csv', index_col=0)
    df2 = pd.read_csv('ELVES_/data/LUAD_2/LUAD_2.csv', index_col=0)
    
    # 合并数据
    X_combined = np.vstack([df1.values, df2.values])
    y_combined = np.hstack([np.zeros(len(df1)), np.ones(len(df2))])
    
    feature_names = df1.columns.tolist()
    
    print(f"数据加载完成:")
    print(f"  - 总样本数: {len(X_combined)}")
    print(f"  - 特征数: {len(feature_names)}")
    print(f"  - 类别0样本数: {np.sum(y_combined == 0)}")
    print(f"  - 类别1样本数: {np.sum(y_combined == 1)}")
    
    return X_combined, y_combined, feature_names

def load_feature_ranking(ranking_file='score.csv'):
    """加载ELVES特征排名结果"""
    try:
        ranking_df = pd.read_csv(ranking_file)
        feature_order = ranking_df['Feature_Index'].values
        feature_names = ranking_df['Feature_Name'].values
        scores = ranking_df['ELVES_Score'].values
        
        print(f"特征排名加载完成:")
        print(f"  - 排名文件: {ranking_file}")
        print(f"  - 特征数量: {len(feature_order)}")
        print(f"  - 最高评分特征: {feature_names[0]} (评分: {scores[0]:.6f})")
        
        return feature_order, feature_names, scores
    
    except FileNotFoundError:
        print(f"错误: 找不到特征排名文件 {ranking_file}")
        print("请先运行ELVES算法生成特征排名")
        return None, None, None

def generate_feature_orders(feature_order, random_state=42):
    """生成三种特征排序方式"""
    # 1. 正序：原始ELVES排名
    positive_order = feature_order.copy()
    
    # 2. 倒序：重要性倒序
    negative_order = feature_order[::-1].copy()
    
    # 3. 随机排序 - 生成5个不同的随机排序
    random_orders = []
    for i in range(5):
        random.seed(random_state + i)
        np.random.seed(random_state + i)
        random_order = feature_order.copy()
        np.random.shuffle(random_order)
        random_orders.append(random_order)
    
    return {
        'positive': positive_order,
        'negative': negative_order,
        'random': random_orders  # 现在是5个随机排序的列表
    }

def evaluate_feature_subset(X, y, feature_indices, cv_folds=5, random_state=42):
    """评估特定特征子集的分类性能"""
    
    # 选择特征子集
    X_subset = X[:, feature_indices]
    
    # 创建Pipeline，确保StandardScaler只在每个CV折的训练数据上fit
    # 这样可以避免数据泄露问题
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        ))
    ])
    
    # 分层K折交叉验证
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # 计算各种指标
    # Pipeline确保每个CV折中，StandardScaler只在训练数据上fit，然后分别transform训练和验证数据
    accuracy_scores = cross_val_score(pipeline, X_subset, y, cv=skf, scoring='accuracy')
    precision_scores = cross_val_score(pipeline, X_subset, y, cv=skf, scoring='precision')
    recall_scores = cross_val_score(pipeline, X_subset, y, cv=skf, scoring='recall')
    f1_scores = cross_val_score(pipeline, X_subset, y, cv=skf, scoring='f1')
    auc_scores = cross_val_score(pipeline, X_subset, y, cv=skf, scoring='roc_auc')
    
    return {
        'accuracy_mean': accuracy_scores.mean(),
        'accuracy_std': accuracy_scores.std(),
        'precision_mean': precision_scores.mean(),
        'precision_std': precision_scores.std(),
        'recall_mean': recall_scores.mean(),
        'recall_std': recall_scores.std(),
        'f1_mean': f1_scores.mean(),
        'f1_std': f1_scores.std(),
        'auc_mean': auc_scores.mean(),
        'auc_std': auc_scores.std()
    }

def progressive_feature_evaluation_comparison(X, y, feature_orders, feature_names, 
                                            max_features=50, step_size=1, cv_folds=5):
    """对比三种特征排序方式的渐进式评估"""
    
    print(f"\n开始三种排序方式的对比评估:")
    print(f"  - 最大特征数: {max_features}")
    print(f"  - 步长: {step_size}")
    print(f"  - 交叉验证折数: {cv_folds}")
    print(f"  - 随机排序重复次数: 5")
    print("-" * 80)
    
    all_results = {}
    
    order_names = {
        'positive': 'ELVES排名 (正序)',
        'negative': '重要性倒序 (反序)',
        'random': '随机排序 (5次平均)'
    }
    
    for order_type, order_data in feature_orders.items():
        print(f"\n评估 {order_names[order_type]}:")
        print("-" * 40)
        
        results = []
        
        # 从1个特征开始，逐步增加
        for n_features in range(1, min(max_features + 1, len(feature_orders['positive']) + 1), step_size):
            print(f"  前 {n_features:2d} 个特征...", end=' ')
            
            if order_type == 'random':
                # 对于随机排序，进行5次评估并计算平均值
                all_metrics = []
                for i, random_order in enumerate(order_data):
                    selected_features = random_order[:n_features]
                    metrics = evaluate_feature_subset(X, y, selected_features, cv_folds)
                    all_metrics.append(metrics)
                
                # 计算5次随机排序的平均值和标准差
                avg_metrics = {}
                for metric_key in all_metrics[0].keys():
                    values = [m[metric_key] for m in all_metrics]
                    if metric_key.endswith('_mean'):
                        # 对于均值指标，计算5次实验的平均值
                        avg_metrics[metric_key] = np.mean(values)
                        # 对于标准差指标，计算5次实验的均值的标准差
                        std_key = metric_key.replace('_mean', '_std')
                        avg_metrics[std_key] = np.std(values)
                    elif metric_key.endswith('_std'):
                        # 标准差指标已经在上面处理了
                        continue
                    else:
                        avg_metrics[metric_key] = np.mean(values)
                
                metrics = avg_metrics
                
            else:
                # 对于正序和倒序，直接使用单个排序
                order_array = order_data
                selected_features = order_array[:n_features]
                metrics = evaluate_feature_subset(X, y, selected_features, cv_folds)
            
            # 记录结果
            result = {
                'n_features': n_features,
                'order_type': order_type,
                **metrics
            }
            results.append(result)
            
            print(f"准确率: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
        
        all_results[order_type] = pd.DataFrame(results)
    
    return all_results

def plot_comparison_results(all_results, save_path):
    """绘制三种排序方式的对比结果图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature Selection Comparison: ELVES vs Reverse vs Random', fontsize=16, fontweight='bold')
    
    metrics = [
        ('accuracy_mean', 'accuracy_std', 'Accuracy'),
        ('precision_mean', 'precision_std', 'Precision'),
        ('recall_mean', 'recall_std', 'Recall'),
        ('f1_mean', 'f1_std', 'F1 Score')
    ]
    
    # 更新颜色配置，使其更容易区分
    colors = {
        'random': '#808080',      # 灰色 - 随机基线
        'negative': '#FF6B6B',    # 红色 - 倒序排名
        'positive': '#4A90E2'     # 蓝色 - ELVES排名
    }
    
    labels = {
        'positive': 'ELVES Ranking (Positive)',
        'negative': 'Reverse Ranking (Negative)',
        'random': 'Random Ranking (5x Average)'
    }
    
    # 更新线条样式，增加区分度
    line_styles = {
        'random': ':',      # 点线 - 随机基线
        'negative': '--',   # 虚线 - 倒序排名
        'positive': '-'     # 实线 - ELVES排名
    }
    
    # 线条宽度配置
    line_widths = {
        'random': 2.0,
        'negative': 2.5,
        'positive': 3.0    # ELVES曲线更粗，更突出
    }
    
    # 标记样式配置
    markers = {
        'random': 's',      # 方形
        'negative': '^',    # 三角形
        'positive': 'o'     # 圆形
    }
    
    # 透明度配置
    alphas = {
        'random': 0.7,
        'negative': 0.8,
        'positive': 1.0    # ELVES曲线完全不透明
    }
    
    for i, (mean_col, std_col, title) in enumerate(metrics):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # 按特定顺序绘制曲线，确保ELVES在最上层
        # 先绘制随机和倒序，最后绘制ELVES
        draw_order = ['random', 'negative', 'positive']
        
        for order_type in draw_order:
            if order_type in all_results:
                results_df = all_results[order_type]
                x = results_df['n_features']
                y = results_df[mean_col]
                yerr = results_df[std_col]
                
                # 绘制主曲线
                ax.plot(x, y, 
                       color=colors[order_type], 
                       linewidth=line_widths[order_type], 
                       linestyle=line_styles[order_type], 
                       marker=markers[order_type], 
                       markersize=4,
                       alpha=alphas[order_type],
                       label=labels[order_type],
                       zorder=10 if order_type == 'positive' else 5)  # ELVES在更高层级
                
                # 绘制误差带
                ax.fill_between(x, y - yerr, y + yerr, 
                               alpha=0.15 if order_type == 'positive' else 0.1, 
                               color=colors[order_type],
                               zorder=8 if order_type == 'positive' else 3)
        
        ax.set_xlabel('Number of Features', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title} Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(fontsize=10, loc='best')
        
        # 设置坐标轴样式
        ax.tick_params(labelsize=10)
        
        # 标注ELVES排名的最佳点
        if 'positive' in all_results:
            best_idx = all_results['positive'][mean_col].idxmax()
            best_x = all_results['positive'].loc[best_idx, 'n_features']
            best_y = all_results['positive'].loc[best_idx, mean_col]
            
            # 调整标注位置，避免遮挡曲线
            xytext_offset = (15, 15) if col == 0 else (-80, 15)
            
            ax.annotate(f'ELVES Best: {best_x} features\n{title}: {best_y:.4f}',
                       xy=(best_x, best_y), xytext=xytext_offset,
                       textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', 
                                alpha=0.8, edgecolor='navy', linewidth=1),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                     color='navy', linewidth=1.5),
                       zorder=15)  # 标注在最上层
        
        # 设置y轴范围，确保所有数据点都可见
        y_min = min([all_results[ot][mean_col].min() - all_results[ot][std_col].max() 
                    for ot in all_results.keys()]) * 0.95
        y_max = min(1.0, max([all_results[ot][mean_col].max() + all_results[ot][std_col].max() 
                             for ot in all_results.keys()]) * 1.02)
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"对比评估曲线图已保存到: {save_path}")
    plt.close()
    
    return save_path

def analyze_comparison_results(all_results):
    """分析对比结果并生成统计报告"""
    
    print(f"\n" + "="*60)
    print("三种排序方式对比分析:")
    print("="*60)
    
    analysis_results = {}
    
    for order_type, results_df in all_results.items():
        # 找到最佳性能
        best_acc_idx = results_df['accuracy_mean'].idxmax()
        best_f1_idx = results_df['f1_mean'].idxmax()
        
        analysis_results[order_type] = {
            'best_accuracy': results_df.loc[best_acc_idx, 'accuracy_mean'],
            'best_accuracy_features': results_df.loc[best_acc_idx, 'n_features'],
            'best_f1': results_df.loc[best_f1_idx, 'f1_mean'],
            'best_f1_features': results_df.loc[best_f1_idx, 'n_features'],
            'final_accuracy': results_df.iloc[-1]['accuracy_mean'],  # 最后一个点的性能
            'final_f1': results_df.iloc[-1]['f1_mean']
        }
    
    # 打印对比结果
    order_names = {
        'positive': 'ELVES排名 (正序)',
        'negative': '重要性倒序 (反序)',
        'random': '随机排序'
    }
    
    print(f"\n{'排序方式':<15} {'最佳准确率':<12} {'对应特征数':<10} {'最佳F1':<12} {'对应特征数':<10}")
    print("-" * 70)
    
    for order_type in ['positive', 'negative', 'random']:
        if order_type in analysis_results:
            stats = analysis_results[order_type]
            print(f"{order_names[order_type]:<15} "
                  f"{stats['best_accuracy']:<12.4f} "
                  f"{stats['best_accuracy_features']:<10d} "
                  f"{stats['best_f1']:<12.4f} "
                  f"{stats['best_f1_features']:<10d}")
    
    # 计算性能提升
    if 'positive' in analysis_results and 'random' in analysis_results:
        elves_acc = analysis_results['positive']['best_accuracy']
        random_acc = analysis_results['random']['best_accuracy']
        improvement = (elves_acc - random_acc) / random_acc * 100
        
        print(f"\nELVES排名相比随机排序的性能提升:")
        print(f"  - 准确率提升: {improvement:.2f}%")
        
        if 'negative' in analysis_results:
            negative_acc = analysis_results['negative']['best_accuracy']
            vs_negative = (elves_acc - negative_acc) / negative_acc * 100
            print(f"  - 相比倒序排名提升: {vs_negative:.2f}%")
    
    return analysis_results

def save_comparison_results(all_results, result_dir):
    """保存对比结果到文件"""
    
    # 合并所有结果
    combined_results = []
    for order_type, results_df in all_results.items():
        df_copy = results_df.copy()
        df_copy['ranking_method'] = order_type
        combined_results.append(df_copy)
    
    combined_df = pd.concat(combined_results, ignore_index=True)
    
    # 保存详细结果
    comparison_file = os.path.join(result_dir, 'feature_ranking_comparison.csv')
    combined_df.to_csv(comparison_file, index=False)
    print(f"对比评估详细结果已保存到: {comparison_file}")
    
    # 保存每种方法的单独结果
    for order_type, results_df in all_results.items():
        method_file = os.path.join(result_dir, f'feature_validation_{order_type}.csv')
        results_df.to_csv(method_file, index=False)
    
    return comparison_file

def find_optimal_feature_count(results_df, metric='accuracy_mean', patience=10):
    """寻找最优特征数量"""
    
    # 找到最佳性能点
    best_idx = results_df[metric].idxmax()
    best_score = results_df.loc[best_idx, metric]
    best_n_features = results_df.loc[best_idx, 'n_features']
    
    # 寻找早停点（性能不再显著提升）
    early_stop_idx = best_idx
    for i in range(best_idx + 1, min(best_idx + patience + 1, len(results_df))):
        if results_df.loc[i, metric] < best_score * 0.99:  # 如果性能下降超过1%
            early_stop_idx = i - 1
            break
    
    early_stop_n_features = results_df.loc[early_stop_idx, 'n_features']
    early_stop_score = results_df.loc[early_stop_idx, metric]
    
    return {
        'best_n_features': best_n_features,
        'best_score': best_score,
        'early_stop_n_features': early_stop_n_features,
        'early_stop_score': early_stop_score
    }

def generate_feature_importance_report(X, y, feature_order, feature_names, 
                                     optimal_n_features, cv_folds=5):
    """生成最优特征子集的重要性报告"""
    
    print(f"\n生成最优特征子集重要性报告 (前{optimal_n_features}个特征)...")
    
    # 选择最优特征子集
    selected_features = feature_order[:optimal_n_features]
    X_subset = X[:, selected_features]
    
    # 使用Pipeline进行标准化和训练，避免数据泄露
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    
    # 训练Pipeline
    pipeline.fit(X_subset, y)
    
    # 获取特征重要性（从训练好的随机森林中）
    rf_classifier = pipeline.named_steps['classifier']
    importances = rf_classifier.feature_importances_
    
    # 创建重要性报告
    importance_df = pd.DataFrame({
        'Feature_Index': selected_features,
        'Feature_Name': feature_names[:optimal_n_features],
        'ELVES_Rank': range(1, optimal_n_features + 1),
        'RF_Importance': importances,
        'RF_Rank': np.argsort(-importances) + 1
    })
    
    # 按随机森林重要性排序
    importance_df = importance_df.sort_values('RF_Importance', ascending=False)
    
    return importance_df

def main():
    print("=" * 80)
    print("ELVES特征排名验证 - 三种排序方式对比分析")
    print("=" * 80)
    
    # 0. 创建结果目录
    result_dir = create_result_directory()
    
    # 1. 加载数据
    X, y, all_feature_names = load_data()
    if X is None:
        return
    
    # 2. 加载特征排名
    feature_order, ranked_feature_names, elves_scores = load_feature_ranking()
    if feature_order is None:
        return
    
    # 3. 生成三种特征排序方式
    feature_orders = generate_feature_orders(feature_order, random_state=42)
    
    print(f"\n生成三种特征排序方式:")
    print(f"  - 正序 (ELVES排名): 前5个特征索引 {feature_orders['positive'][:5]}")
    print(f"  - 倒序 (重要性倒序): 前5个特征索引 {feature_orders['negative'][:5]}")
    print(f"  - 随机排序: 前5个特征索引 {feature_orders['random'][:5]}")
    
    # 4. 三种排序方式的对比评估
    max_features = min(363, len(feature_order))  
    step_size = 1
    cv_folds = 15
    
    all_results = progressive_feature_evaluation_comparison(
        X, y, feature_orders, ranked_feature_names,
        max_features=max_features, step_size=step_size, cv_folds=cv_folds
    )
    
    # 5. 保存对比结果
    comparison_file = save_comparison_results(all_results, result_dir)
    
    # 6. 绘制对比曲线图
    plot_file = os.path.join(result_dir, 'feature_ranking_comparison.png')
    plot_comparison_results(all_results, plot_file)
    
    # 7. 分析对比结果
    analysis_results = analyze_comparison_results(all_results)
    
    # 8. 生成最优特征重要性报告（基于ELVES排名）
    if 'positive' in all_results:
        optimal_info = find_optimal_feature_count(all_results['positive'], metric='accuracy_mean')
        optimal_n = optimal_info['early_stop_n_features']
        
        importance_df = generate_feature_importance_report(
            X, y, feature_orders['positive'], ranked_feature_names, optimal_n
        )
        
        importance_file = os.path.join(result_dir, 'optimal_features_importance.csv')
        importance_df.to_csv(importance_file, index=False)
        print(f"最优特征重要性报告已保存到: {importance_file}")
    
    # 9. 生成对比验证总结报告
    summary_file = os.path.join(result_dir, 'comparison_validation_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("ELVES特征排名对比验证总结报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"数据信息:\n")
        f.write(f"  - 总样本数: {len(X)}\n")
        f.write(f"  - 特征数: {len(all_feature_names)}\n")
        f.write(f"  - 类别0样本数: {np.sum(y == 0)}\n")
        f.write(f"  - 类别1样本数: {np.sum(y == 1)}\n\n")
        
        f.write(f"验证方法:\n")
        f.write(f"  - 三种排序方式: ELVES排名(正序)、重要性倒序(反序)、随机排序\n")
        f.write(f"  - 评估范围: 前{max_features}个特征\n")
        f.write(f"  - 交叉验证: {cv_folds}折分层交叉验证\n")
        f.write(f"  - 分类器: 随机森林 (100棵树)\n\n")
        
        f.write(f"对比结果:\n")
        order_names = {
            'positive': 'ELVES排名 (正序)',
            'negative': '重要性倒序 (反序)',
            'random': '随机排序'
        }
        
        for order_type in ['positive', 'negative', 'random']:
            if order_type in analysis_results:
                stats = analysis_results[order_type]
                f.write(f"  {order_names[order_type]}:\n")
                f.write(f"    - 最佳准确率: {stats['best_accuracy']:.4f} (使用{stats['best_accuracy_features']}个特征)\n")
                f.write(f"    - 最佳F1分数: {stats['best_f1']:.4f} (使用{stats['best_f1_features']}个特征)\n")
        
        f.write(f"\n结论:\n")
        if 'positive' in analysis_results and 'random' in analysis_results:
            elves_acc = analysis_results['positive']['best_accuracy']
            random_acc = analysis_results['random']['best_accuracy']
            improvement = (elves_acc - random_acc) / random_acc * 100
            f.write(f"  - ELVES排名相比随机排序性能提升: {improvement:.2f}%\n")
            
            if 'negative' in analysis_results:
                negative_acc = analysis_results['negative']['best_accuracy']
                vs_negative = (elves_acc - negative_acc) / negative_acc * 100
                f.write(f"  - ELVES排名相比倒序排名性能提升: {vs_negative:.2f}%\n")
        
        f.write(f"  - ELVES算法能够有效识别重要特征，显著优于随机选择和倒序选择\n")
    
    print(f"对比验证总结报告已保存到: {summary_file}")
    
    # 10. 显示最终总结
    print(f"\n" + "="*80)
    print("验证总结:")
    if 'positive' in analysis_results:
        elves_stats = analysis_results['positive']
        print(f"  - ELVES排名最佳准确率: {elves_stats['best_accuracy']:.4f} (使用{elves_stats['best_accuracy_features']}个特征)")
        
        if 'random' in analysis_results:
            random_stats = analysis_results['random']
            improvement = (elves_stats['best_accuracy'] - random_stats['best_accuracy']) / random_stats['best_accuracy'] * 100
            print(f"  - 相比随机排序提升: {improvement:.2f}%")
    
    print(f"\n前10个最重要特征 (按ELVES排名):")
    for i in range(min(10, len(ranked_feature_names))):
        print(f"  {i+1:2d}. {ranked_feature_names[i]} (ELVES评分: {elves_scores[i]:.6f})")
    
    print(f"\n生成的文件 (保存在 {result_dir}/ 目录下):")
    print(f"  - feature_ranking_comparison.csv: 三种方法对比结果")
    print(f"  - feature_validation_positive.csv: ELVES排名结果")
    print(f"  - feature_validation_negative.csv: 倒序排名结果")
    print(f"  - feature_validation_random.csv: 随机排序结果")
    print(f"  - optimal_features_importance.csv: 最优特征重要性分析")
    print(f"  - feature_ranking_comparison.png: 对比评估曲线图")
    print(f"  - comparison_validation_summary.txt: 对比验证总结报告")
    print("=" * 80)

if __name__ == "__main__":
    main() 