#!/usr/bin/env python3

import os
import networkx as nx
import pandas as pd
import numpy as np
import traceback
import time
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

print("通路网络MCE计算程序 (优化版)")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局样本MCE数据
SAMPLE_MCE_DATA = {}

def load_sample_mce_data(sample_mce_file):
    """
    加载样本MCE汇总数据
    """
    global SAMPLE_MCE_DATA
    try:
        if os.path.exists(sample_mce_file):
            df = pd.read_csv(sample_mce_file)
            # 创建样本名到归一化MCE的映射
            SAMPLE_MCE_DATA = dict(zip(df['sample_name'], df['normalized_mce']))
            logger.info(f"成功加载 {len(SAMPLE_MCE_DATA)} 个样本的MCE数据")
            return True
        else:
            logger.error(f"样本MCE汇总文件不存在: {sample_mce_file}")
            return False
    except Exception as e:
        logger.error(f"加载样本MCE数据时出错: {str(e)}")
        return False

def ensure_directory(directory_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"创建目录: {directory_path}")

class PathwayDataCache:
    """通路数据缓存类，避免重复加载"""
    def __init__(self):
        self.pathway_cache = {}
        self.sample_data_cache = {}
    
    def get_pathway_data(self, pathway_file):
        """获取缓存的通路数据"""
        if pathway_file not in self.pathway_cache:
            try:
                G = nx.read_graphml(pathway_file)
                # 预计算邻接矩阵
                A = nx.to_numpy_array(G)
                np.fill_diagonal(A, 1)
                
                self.pathway_cache[pathway_file] = {
                    'nodes': list(G.nodes()),
                    'adjacency_matrix': A,
                    'node_count': G.number_of_nodes(),
                    'edge_count': G.number_of_edges()
                }
            except Exception as e:
                logger.error(f"加载通路文件 {pathway_file} 失败: {str(e)}")
                return None
        
        return self.pathway_cache[pathway_file]

# 全局缓存实例
pathway_cache = PathwayDataCache()

def load_sample_vectors_and_matrix_optimized(sample_output_dir, sample_name):
    """
    优化的样本数据加载函数，支持缓存
    """
    cache_key = f"{sample_output_dir}_{sample_name}"
    
    if cache_key in pathway_cache.sample_data_cache:
        return pathway_cache.sample_data_cache[cache_key]
    
    try:
        # 使用更高效的数据类型
        pi_file = os.path.join(sample_output_dir, f"{sample_name}_pi_vector.csv")
        sparse_p_file = os.path.join(sample_output_dir, f"{sample_name}_P_matrix_sparse.csv")
        mapping_file = os.path.join(sample_output_dir, f"{sample_name}_node_mapping.csv")
        
        # 检查文件存在性
        for file_path in [pi_file, sparse_p_file, mapping_file]:
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return None, None, None
        
        # 并行读取文件
        pi_df = pd.read_csv(pi_file, dtype={'node_id': 'int32', 'pi': 'float32'})
        sparse_p_df = pd.read_csv(sparse_p_file, dtype={
            'node_i_id': 'int32', 
            'node_j_id': 'int32', 
            'pij_value': 'float32'
        })
        node_mapping = pd.read_csv(mapping_file, dtype={'node_id': 'int32', 'vector_index': 'int32'})
        
        # 转换为字典以加速查找
        pi_dict = dict(zip(pi_df['node_id'].values, pi_df['pi'].values))
        
        # 创建稀疏矩阵索引
        sparse_p_dict = {}
        for row in sparse_p_df.itertuples(index=False):
            sparse_p_dict[(row.node_i_id, row.node_j_id)] = row.pij_value
        
        result = (pi_dict, sparse_p_dict, node_mapping)
        pathway_cache.sample_data_cache[cache_key] = result
        
        logger.info(f"加载样本数据: {len(pi_dict)} 个节点, {len(sparse_p_dict)} 个非零元素")
        return result
        
    except Exception as e:
        logger.error(f"加载样本向量和矩阵时出错: {str(e)}")
        return None, None, None

def extract_pathway_submatrix_vectorized(pathway_nodes, pi_dict, sparse_p_dict, node_mapping):
    """
    向量化的子矩阵提取函数
    """
    try:
        # 快速类型转换和筛选
        pathway_nodes_int = []
        for node in pathway_nodes:
            try:
                pathway_nodes_int.append(int(node))
            except (ValueError, TypeError):
                pathway_nodes_int.append(node)
        
        # 使用集合运算快速找到有效节点
        available_nodes = set(pi_dict.keys())
        pathway_nodes_set = set(pathway_nodes_int)
        valid_nodes = list(pathway_nodes_set & available_nodes)
        
        if len(valid_nodes) <= 1:
            return None, None, None, None
        
        # 快速创建索引映射
        node_to_index = {node: i for i, node in enumerate(valid_nodes)}
        n = len(valid_nodes)
        
        # 向量化提取pi向量
        pathway_pi = np.array([pi_dict[node] for node in valid_nodes], dtype=np.float32)
        
        # 向量化构建P矩阵
        pathway_P = np.zeros((n, n), dtype=np.float32)
        
        # 批量处理稀疏矩阵元素
        for (node_i, node_j), pij_value in sparse_p_dict.items():
            if node_i in node_to_index and node_j in node_to_index:
                i = node_to_index[node_i]
                j = node_to_index[node_j]
                pathway_P[i, j] = pij_value
        
        # 找到对应的原始节点
        original_valid_nodes = [str(node) for node in valid_nodes]
        
        return pathway_pi, pathway_P, valid_nodes, original_valid_nodes
        
    except Exception as e:
        logger.error(f"提取通路子矩阵时出错: {str(e)}")
        return None, None, None, None

def calculate_pathway_mce_vectorized(pi, P, A):
    """
    向量化的MCE计算函数
    """
    try:
        # 避免数值问题
        pi = np.maximum(pi, 1e-10)
        P = np.maximum(P, 1e-10)
        
        # 第一部分: -∑πi ∑(j∈N(i)) pij log pij (向量化)
        # 只计算有邻居关系的部分
        neighbor_mask = A > 0
        P_masked = np.where(neighbor_mask, P, 0)
        
        # 向量化计算对数项
        log_P = np.where(P_masked > 1e-10, np.log2(P_masked), 0)
        neighbor_entropy = np.sum(P_masked * log_P, axis=1)
        first_part = -np.sum(pi * neighbor_entropy)
        
        # 第二部分: -∑πi log πi (向量化)
        second_part = -np.sum(pi * np.log2(pi))
        
        mce_value = first_part + second_part
        
        return 0.0 if np.isnan(mce_value) else float(mce_value)
        
    except Exception as e:
        logger.error(f"计算MCE时出错: {str(e)}")
        return 0.0

def normalize_pi_vector_fast(pi):
    """快速pi向量归一化"""
    pi_sum = np.sum(pi)
    return np.ones_like(pi) / len(pi) if pi_sum == 0 else pi / pi_sum

def row_normalize_p_matrix_fast(P):
    """快速P矩阵行归一化"""
    row_sums = np.sum(P, axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1)  # 避免除零
    return P / row_sums

def calculate_mce_ratio(pathway_normalized_mce, sample_name):
    """
    计算MCE比率 (通路归一化MCE / 样本归一化MCE)
    """
    global SAMPLE_MCE_DATA
    
    if sample_name in SAMPLE_MCE_DATA:
        sample_normalized_mce = SAMPLE_MCE_DATA[sample_name]
        if sample_normalized_mce is not None and sample_normalized_mce > 0:
            return pathway_normalized_mce / sample_normalized_mce
        else:
            logger.warning(f"样本 {sample_name} 的归一化MCE为零或空值")
            return None
    else:
        logger.warning(f"未找到样本 {sample_name} 的MCE数据")
        return None

def process_single_pathway(args):
    """
    处理单个通路的函数（用于多进程）
    """
    try:
        (pathway_file, pathway_name, sample_name, 
         pi_dict, sparse_p_dict, node_mapping) = args
        
        # 获取缓存的通路数据
        pathway_data = pathway_cache.get_pathway_data(pathway_file)
        if pathway_data is None:
            return create_error_result(pathway_name, sample_name, "Failed to load pathway")
        
        if pathway_data['node_count'] <= 1:
            return create_error_result(pathway_name, sample_name, "Too few nodes")
        
        # 提取子矩阵
        pathway_pi, pathway_P, valid_nodes, original_valid_nodes = extract_pathway_submatrix_vectorized(
            pathway_data['nodes'], pi_dict, sparse_p_dict, node_mapping
        )
        
        if pathway_pi is None:
            return create_error_result(pathway_name, sample_name, "Failed to extract submatrix")
        
        # 构建有效节点的邻接矩阵
        try:
            # 创建节点索引映射
            node_to_original_index = {node: i for i, node in enumerate(pathway_data['nodes'])}
            valid_indices = [node_to_original_index[node] for node in original_valid_nodes 
                           if node in node_to_original_index]
            
            if len(valid_indices) != len(valid_nodes):
                return create_error_result(pathway_name, sample_name, "Index mismatch")
            
            # 提取子邻接矩阵
            A_valid = pathway_data['adjacency_matrix'][np.ix_(valid_indices, valid_indices)]
            
        except Exception as e:
            return create_error_result(pathway_name, sample_name, f"Adjacency matrix error: {str(e)}")
        
        # 快速归一化
        pathway_pi_normalized = normalize_pi_vector_fast(pathway_pi)
        pathway_P_normalized = row_normalize_p_matrix_fast(pathway_P)
        
        # 向量化MCE计算
        mce_value = calculate_pathway_mce_vectorized(
            pathway_pi_normalized, pathway_P_normalized, A_valid
        )
        
        # 计算最大MCE
        m_valid = np.count_nonzero(A_valid)
        max_mce = np.log2(m_valid) if m_valid > 0 else 0.0
        normalized_mce = mce_value / max_mce if max_mce > 0 else 0.0
        
        # 计算MCE比率
        mce_ratio = calculate_mce_ratio(normalized_mce, sample_name)
        
        return {
            'sample_name': sample_name,
            'pathway_name': pathway_name,
            'mce': mce_value,
            'max_mce': max_mce,
            'normalized_mce': normalized_mce,
            'mce_ratio': mce_ratio,
            'nodes_count': len(valid_nodes),
            'edges_count': m_valid,
            'original_nodes_count': pathway_data['node_count'],
            'original_edges_count': pathway_data['edge_count'],
            'success': True
        }
        
    except Exception as e:
        logger.error(f"处理通路 {pathway_name} 时出错: {str(e)}")
        return create_error_result(pathway_name, sample_name, str(e))

def create_error_result(pathway_name, sample_name, error_msg):
    """创建错误结果"""
    return {
        'sample_name': sample_name,
        'pathway_name': pathway_name,
        'mce': None,
        'max_mce': None,
        'normalized_mce': None,
        'mce_ratio': None,
        'nodes_count': 0,
        'edges_count': 0,
        'original_nodes_count': 0,
        'original_edges_count': 0,
        'success': False,
        'error': error_msg
    }

def get_processed_pathways(sample_results_file):
    """
    获取已处理的通路列表
    
    参数:
    sample_results_file -- 样本结果文件路径
    
    返回:
    已处理的通路名称集合
    """
    if not os.path.exists(sample_results_file):
        return set()
    
    try:
        df = pd.read_csv(sample_results_file)
        processed_pathways = set(df['pathway_name'].tolist())
        print(f"发现已处理的通路: {len(processed_pathways)} 个")
        return processed_pathways
    except Exception as e:
        print(f"读取已处理通路时出错: {str(e)}")
        return set()

def save_batch_results(results, sample_results_file, summary_file, batch_mode=True):
    """
    批量保存结果（提高I/O效率）
    """
    try:
        if not results:
            return
        
        # 明确指定列顺序，包含新增的mce_ratio列
        column_order = [
            'sample_name', 'pathway_name', 'mce', 'max_mce', 'normalized_mce', 'mce_ratio',
            'nodes_count', 'edges_count', 'original_nodes_count', 'original_edges_count',
            'success', 'error'
        ]
        
        results_df = pd.DataFrame(results)
        
        # 确保所有列都存在，如果不存在就用None填充
        for col in column_order:
            if col not in results_df.columns:
                results_df[col] = None
        
        # 按指定顺序重新排列列
        results_df = results_df[column_order]
        
        # 保存到样本结果文件
        if os.path.exists(sample_results_file):
            results_df.to_csv(sample_results_file, mode='a', header=False, index=False)
        else:
            results_df.to_csv(sample_results_file, mode='w', header=True, index=False)
        
        # 保存到汇总结果文件
        if os.path.exists(summary_file):
            results_df.to_csv(summary_file, mode='a', header=False, index=False)
        else:
            results_df.to_csv(summary_file, mode='w', header=True, index=False)
        
        successful_count = sum(1 for r in results if r['success'])
        logger.info(f"批量保存 {len(results)} 个结果 (成功: {successful_count})")
        
    except Exception as e:
        logger.error(f"批量保存结果时出错: {str(e)}")

def get_pathway_files(pathways_dir, sample_name):
    """
    获取指定样本的所有通路网络文件
    """
    try:
        sample_pathway_dir = os.path.join(pathways_dir, sample_name)
        if not os.path.exists(sample_pathway_dir):
            logger.error(f"样本通路目录不存在: {sample_pathway_dir}")
            return []
        
        pathway_files = [f for f in os.listdir(sample_pathway_dir) if f.endswith('.graphml')]
        logger.info(f"样本 {sample_name} 有 {len(pathway_files)} 个通路文件")
        
        return [(os.path.join(sample_pathway_dir, f), os.path.splitext(f)[0]) for f in pathway_files]
        
    except Exception as e:
        logger.error(f"获取通路文件时出错: {str(e)}")
        return []

def process_sample_pathways_parallel(sample_name, sample_vectors_dir, pathways_dir, 
                                   output_dir, max_workers=None, batch_size=50):
    """
    并行处理单个样本的所有通路网络
    """
    try:
        logger.info(f"开始并行处理样本: {sample_name}")
        
        ensure_directory(output_dir)
        
        sample_results_file = os.path.join(output_dir, f"{sample_name}_pathway_mce_results.csv")
        summary_file = os.path.join(output_dir, "all_pathway_mce_results.csv")
        
        processed_pathways = get_processed_pathways(sample_results_file)
        
        # 加载样本数据
        sample_output_dir = os.path.join(sample_vectors_dir, sample_name)
        pi_dict, sparse_p_dict, node_mapping = load_sample_vectors_and_matrix_optimized(
            sample_output_dir, sample_name
        )
        
        if pi_dict is None:
            logger.error(f"无法加载样本 {sample_name} 的向量数据")
            return {'total': 0, 'successful': 0, 'failed': 0, 'skipped': 0}
        
        pathway_files = get_pathway_files(pathways_dir, sample_name)
        if not pathway_files:
            return {'total': 0, 'successful': 0, 'failed': 0, 'skipped': 0}
        
        # 筛选未处理的通路
        unprocessed_pathways = [
            (pathway_file, pathway_name) for pathway_file, pathway_name in pathway_files
            if pathway_name not in processed_pathways
        ]
        
        if not unprocessed_pathways:
            logger.info(f"样本 {sample_name} 的所有通路已处理完成")
            return {
                'total': len(pathway_files), 
                'successful': 0, 
                'failed': 0, 
                'skipped': len(pathway_files)
            }
        
        logger.info(f"需要处理 {len(unprocessed_pathways)} 个通路")
        
        # 准备并行处理的参数
        process_args = [
            (pathway_file, pathway_name, sample_name, pi_dict, sparse_p_dict, node_mapping)
            for pathway_file, pathway_name in unprocessed_pathways
        ]
        
        # 动态设置工作进程数
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(unprocessed_pathways), 8)
        
        successful_count = 0
        failed_count = 0
        results_batch = []
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_pathway = {
                executor.submit(process_single_pathway, args): args[1] 
                for args in process_args
            }
            
            # 处理完成的任务
            for future in tqdm(as_completed(future_to_pathway), 
                             total=len(future_to_pathway), 
                             desc=f"并行处理{sample_name}"):
                
                pathway_name = future_to_pathway[future]
                try:
                    result = future.result()
                    results_batch.append(result)
                    
                    if result['success']:
                        successful_count += 1
                    else:
                        failed_count += 1
                    
                    # 批量保存结果
                    if len(results_batch) >= batch_size:
                        save_batch_results(results_batch, sample_results_file, summary_file)
                        results_batch = []
                    
                except Exception as e:
                    logger.error(f"处理通路 {pathway_name} 时出错: {str(e)}")
                    failed_count += 1
        
        # 保存剩余的结果
        if results_batch:
            save_batch_results(results_batch, sample_results_file, summary_file)
        
        total_pathways = len(pathway_files)
        skipped_count = len(processed_pathways)
        
        logger.info(f"样本 {sample_name} 并行处理完成:")
        logger.info(f"  总通路数: {total_pathways}")
        logger.info(f"  成功处理: {successful_count}")
        logger.info(f"  处理失败: {failed_count}")
        logger.info(f"  跳过已处理: {skipped_count}")
        
        return {
            'total': total_pathways,
            'successful': successful_count,
            'failed': failed_count,
            'skipped': skipped_count
        }
        
    except Exception as e:
        logger.error(f"并行处理样本 {sample_name} 时出错: {str(e)}")
        return {'total': 0, 'successful': 0, 'failed': 0, 'skipped': 0}

def main():
    """
    主函数（并行优化版）
    """
    print("=" * 60)
    print("通路网络MCE计算程序 (并行优化版)")
    print("=" * 60)
    
    # 配置路径
    sample_vectors_base_dir = "/root/autodl-tmp/Sample_LUSC_MCE_Vectors_1"
    pathways_dir = "/root/autodl-tmp/Final_pathway_LUSC"
    output_dir = "/root/autodl-tmp/Pathway_MCE_Results_LUSC"
    sample_mce_file = "/root/autodl-tmp/Sample_LUSC_MCE_Vectors_1/sample_LUSC_mce_summary.csv"
    
    # 加载样本MCE数据
    print("加载样本MCE汇总数据...")
    if not load_sample_mce_data(sample_mce_file):
        logger.error("无法加载样本MCE数据，程序退出")
        return
    
    # 性能配置
    max_workers = min(mp.cpu_count(), 8)  # 限制最大进程数
    batch_size = 100  # 批处理大小
    
    print(f"使用 {max_workers} 个并行进程")
    print(f"批处理大小: {batch_size}")
    
    # 检查输入目录
    for directory, name in [(sample_vectors_base_dir, "样本向量目录"), 
                           (pathways_dir, "通路目录")]:
        if not os.path.exists(directory):
            logger.error(f"{name}不存在: {directory}")
            return
    
    ensure_directory(output_dir)
    
    # 获取样本列表
    sample_names = [d for d in os.listdir(sample_vectors_base_dir) 
                   if os.path.isdir(os.path.join(sample_vectors_base_dir, d))]
    
    if not sample_names:
        logger.error("未找到任何样本目录")
        return
    
    logger.info(f"找到 {len(sample_names)} 个样本")
    
    # 处理统计
    total_stats = {'total': 0, 'successful': 0, 'failed': 0, 'skipped': 0}
    start_time = time.time()
    
    # 并行处理所有样本
    for i, sample_name in enumerate(sample_names, 1):
        logger.info(f"[{i}/{len(sample_names)}] 开始并行处理样本: {sample_name}")
        
        sample_stats = process_sample_pathways_parallel(
            sample_name, sample_vectors_base_dir, pathways_dir, output_dir,
            max_workers=max_workers, batch_size=batch_size
        )
        
        # 累计统计
        for key in total_stats:
            total_stats[key] += sample_stats[key]
        
        # 每处理3个样本输出一次进度
        if i % 3 == 0 or i == len(sample_names):
            elapsed_time = time.time() - start_time
            avg_time_per_sample = elapsed_time / i
            estimated_remaining = avg_time_per_sample * (len(sample_names) - i)
            
            logger.info(f"进度统计 [{i}/{len(sample_names)}]:")
            logger.info(f"  总通路数: {total_stats['total']}")
            logger.info(f"  成功处理: {total_stats['successful']}")
            logger.info(f"  处理失败: {total_stats['failed']}")
            logger.info(f"  跳过已处理: {total_stats['skipped']}")
            logger.info(f"  已用时间: {elapsed_time/60:.1f} 分钟")
            logger.info(f"  预计剩余: {estimated_remaining/60:.1f} 分钟")
            logger.info(f"  平均处理速度: {total_stats['successful']/(elapsed_time/60):.1f} 通路/分钟")
    
    # 最终统计
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("并行处理完成!")
    print(f"处理样本数: {len(sample_names)}")
    print(f"总用时: {total_time/60:.1f} 分钟")
    print(f"总通路数: {total_stats['total']}")
    print(f"成功处理: {total_stats['successful']}")
    print(f"处理失败: {total_stats['failed']}")
    print(f"跳过已处理: {total_stats['skipped']}")
    print(f"成功率: {total_stats['successful']/max(total_stats['total']-total_stats['skipped'], 1)*100:.1f}%")
    print(f"平均处理速度: {total_stats['successful']/(total_time/60):.1f} 通路/分钟")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main() 