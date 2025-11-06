#!/usr/bin/env python3

import os
import networkx as nx
import pandas as pd
from collections import defaultdict
import csv
import time
import numpy as np
from tqdm import tqdm
from functools import partial
import concurrent.futures
import sys
import subprocess
import torch
import traceback
import resource
import signal

print("使用PyTorch和GPU加速实现")

# 尝试导入CuPy用于GPU加速
try:
    import cupy as cp
    HAS_CUPY = True
    print("使用CuPy进行GPU加速MCE计算")
except ImportError:
    HAS_CUPY = False
    print("未找到CuPy库，将使用CPU计算MCE")

def ensure_directory(directory_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"创建目录: {directory_path}")

# 检查GPU是否可用
use_gpu = torch.cuda.is_available()
if use_gpu:
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    # 预热GPU
    dummy_tensor = torch.ones(1, device='cuda')
    del dummy_tensor
    torch.cuda.empty_cache()
else:
    print("GPU不可用，将使用CPU")

# GPU版任务管理器以保持GPU活跃并防止自动降频
class ContinuousGPUTaskManager:
    def __init__(self, num_streams=4):
        if not use_gpu:
            self.running = False
            return
        self.num_streams = num_streams
        self.running = False
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.task_tensors = []
        
    def start_background_task(self):
        """在后台启动持续性GPU任务，以保持GPU活跃"""
        if not use_gpu:
            return
        try:
            import threading
            self.running = True
            self.thread = threading.Thread(target=self._run_continuous_task)
            self.thread.daemon = True  # 设置为守护线程，这样当主程序退出时，这个线程也会退出
            self.thread.start()
            print("GPU任务管理器已启动")
        except Exception as e:
            print(f"启动GPU任务管理器失败: {e}")
            self.running = False
            # 不将use_gpu设置为False，只是不使用后台任务
            
    def _run_continuous_task(self):
        """保持GPU活跃的连续任务"""
        if not use_gpu:
            return
        try:
            # 创建一些小矩阵并持续进行简单计算
            for i in range(self.num_streams):
                self.task_tensors.append(torch.ones(128, 128, device='cuda'))
            
            while self.running:
                for i, stream in enumerate(self.streams):
                    with torch.cuda.stream(stream):
                        # 简单矩阵操作，保持GPU活跃但不消耗太多资源
                        self.task_tensors[i] = torch.matmul(self.task_tensors[i], self.task_tensors[i]) * 0.999
                        # 添加同步点以避免过度堆积任务
                        stream.synchronize()
                # 短暂休眠以避免过度使用CPU
                time.sleep(0.05)
        except Exception as e:
            print(f"GPU后台任务出错: {e}")
            self.running = False
            
    def stop(self):
        """停止后台任务"""
        if not self.running:
            return
        self.running = False
        if hasattr(self, 'thread'):
            try:
                self.thread.join(timeout=1.0)  # 等待线程结束，超时1秒
            except Exception:
                pass  # 忽略线程终止错误
        # 清理资源
        if hasattr(self, 'task_tensors'):
            self.task_tensors = []
        if use_gpu:
            torch.cuda.empty_cache()
            
# 判断是否内存使用过高
def is_memory_critical():
    """检查内存使用是否超过阈值"""
    if use_gpu:
        # GPU内存使用
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # 转换为GB
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # 转换为GB
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # 转换为GB
        memory_usage_percent = (memory_reserved / total_memory) * 100
        
        if memory_usage_percent > 75:  # 超过75%就认为内存使用过高
            print(f"GPU内存使用率: {memory_usage_percent:.2f}% ({memory_reserved:.2f}GB/{total_memory:.2f}GB)")
            return True
    
    # CPU内存检查
    try:
        total_memory = os.popen('free -g').readlines()[1].split()[1]
        used_memory = os.popen('free -g').readlines()[1].split()[2]
        memory_usage_percent = (int(used_memory) / int(total_memory)) * 100
        
        if memory_usage_percent > 85:  # 超过85%就认为内存使用过高
            print(f"系统内存使用率: {memory_usage_percent:.2f}% ({used_memory}GB/{total_memory}GB)")
            return True
    except:
        pass
    
    return False

# 将NetworkX图转换为PyTorch张量
def compute_adjacency_matrix_gpu(G):
    """
    将NetworkX图转换为PyTorch张量格式的邻接矩阵
    
    参数:
    G: NetworkX图
    
    返回:
    邻接矩阵的PyTorch张量表示，成功标志
    """
    try:
        if not use_gpu:
            # 如果不使用GPU，则使用NetworkX的邻接矩阵
            return nx.to_numpy_array(G), True
        
        # 获取节点数
        n = G.number_of_nodes()
        
        # 节点到索引的映射
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        
        # 对于任何大小的图，直接使用稠密表示避免稀疏矩阵操作问题
        print(f"处理图 ({n} 个节点), 使用稠密矩阵表示")
        
        # 创建邻接矩阵
        adj = torch.zeros((n, n), device='cuda')
        
        # 获取边数量
        edge_count = G.number_of_edges()
        if edge_count == 0:
            print("警告: 图中没有边，返回全零矩阵")
            return adj, True
        
        # 批处理边以减少内存占用
        print(f"开始处理 {edge_count} 条边...")
        batch_size = 100000  # 每次处理的边数量
        edges = list(G.edges(data=True))
        
        for i in range(0, len(edges), batch_size):
            # 清理内存以防占用过高
            if i > 0 and i % (5 * batch_size) == 0:
                torch.cuda.empty_cache()
                
            end_idx = min(i + batch_size, len(edges))
            batch_edges = edges[i:end_idx]
            
            # 为批处理创建临时张量
            edge_indices = torch.zeros((len(batch_edges), 2), dtype=torch.long)
            edge_values = torch.ones(len(batch_edges))
            
            for j, (u, v, data) in enumerate(batch_edges):
                idx_u = node_to_idx[u]
                idx_v = node_to_idx[v]
                edge_indices[j, 0] = idx_u
                edge_indices[j, 1] = idx_v
                # 如果边有权重，则使用权重，否则使用1
                edge_values[j] = data.get('weight', 1.0)
            
            # 将临时张量传输到GPU
            edge_indices = edge_indices.cuda()
            edge_values = edge_values.cuda()
            
            # 直接在邻接矩阵上设置值
            adj[edge_indices[:, 0], edge_indices[:, 1]] = edge_values
            
            # 清理临时张量
            del edge_indices
            del edge_values
            
            if (i // batch_size) % 5 == 0:
                print(f"已处理 {i+len(batch_edges)}/{edge_count} 条边 ({((i+len(batch_edges))/edge_count)*100:.1f}%)")
                
        print("邻接矩阵创建完成")
        return adj, True
            
    except Exception as e:
        print(f"计算邻接矩阵时出错: {e}")
        traceback.print_exc()
        return None, False

# GPU版本的最短路径计算
def compute_shortest_paths_gpu(adj_matrix, source_idx):
    """
    使用GPU计算从源节点到所有其他节点的最短路径
    
    参数:
    adj_matrix: 邻接矩阵的PyTorch张量表示
    source_idx: 源节点的索引
    
    返回:
    到所有节点的最短路径距离，最短路径的前驱节点
    """
    if not use_gpu:
        # 如果不使用GPU，则转为numpy执行
        adj_np = adj_matrix.numpy()
        
        # 使用Dijkstra算法
        n = adj_np.shape[0]
        dist = np.full(n, np.inf)
        dist[source_idx] = 0
        pred = np.full(n, -1, dtype=np.int32)
        visited = np.zeros(n, dtype=bool)
        
        for _ in range(n):
            # 找到未访问的最小距离节点
            u = -1
            min_dist = np.inf
            for i in range(n):
                if not visited[i] and dist[i] < min_dist:
                    min_dist = dist[i]
                    u = i
            
            if u == -1 or np.isinf(min_dist):
                break
            
            visited[u] = True
            
            # 更新邻居的距离
            for v in range(n):
                if adj_np[u, v] > 0 and not visited[v]:
                    new_dist = dist[u] + 1  # 使用单位距离
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        pred[v] = u
        
        return dist, pred
    
    # GPU实现
    n = adj_matrix.shape[0]
    device = adj_matrix.device
    
    # 初始化距离和前驱数组
    dist = torch.full((n,), float('inf'), device=device)
    dist[source_idx] = 0
    pred = torch.full((n,), -1, dtype=torch.int64, device=device)
    
    # 创建邻接关系的二值掩码（1表示有边，0表示无边）
    mask = (adj_matrix > 0).float()
    
    # 使用BFS进行最短路径计算
    frontier = torch.zeros(n, dtype=torch.bool, device=device)
    frontier[source_idx] = True
    
    for _ in range(n):  # 最大迭代次数为n
        if not frontier.any():
            break
        
        # 计算新的前沿节点
        new_dist = dist.unsqueeze(1).expand(-1, n)
        new_dist = torch.where(frontier.unsqueeze(1), new_dist + 1, torch.tensor(float('inf'), device=device))
        
        # 使用掩码过滤掉不存在的边
        new_dist = torch.where(mask > 0, new_dist, torch.tensor(float('inf'), device=device))
        
        # 找到能够提供更短路径的节点
        improved = new_dist < dist.unsqueeze(0)
        
        # 更新距离和前驱
        if improved.any():
            # 对于每个目标节点，找到提供最短路径的源节点
            best_source = torch.argmin(torch.where(improved, new_dist, torch.tensor(float('inf'), device=device)), dim=0)
            
            # 创建更新掩码
            update_mask = improved.any(dim=0)
            
            # 更新距离和前驱
            new_dists = torch.gather(new_dist, 0, best_source.unsqueeze(0)).squeeze(0)
            dist = torch.where(update_mask, new_dists, dist)
            pred = torch.where(update_mask, best_source, pred)
            
            # 更新前沿
            frontier = update_mask & ~frontier
        else:
            break
    
    return dist, pred

# GPU版本的补充代价计算函数
def calculate_complement_cost_gpu(adj_matrix, B_nodes_idx, node_idx, node_to_idx, idx_to_node):
    """
    使用GPU计算将节点作为中心节点的补充代价
    
    参数:
    adj_matrix: 邻接矩阵的PyTorch张量表示
    B_nodes_idx: 子网络节点的索引集合
    node_idx: 候选中心节点的索引
    node_to_idx: 节点到索引的映射
    idx_to_node: 索引到节点的映射
    
    返回:
    cost_out: 正向路径补充代价
    cost_in: 反向路径补充代价
    nodes_to_add: 需要添加的节点
    edges_to_add: 需要添加的边
    """
    try:
        n = adj_matrix.shape[0]
        device = adj_matrix.device if use_gpu else 'cpu'
        
        # 创建B的邻接矩阵（子网络）
        B_adj = torch.zeros_like(adj_matrix)
        for i in B_nodes_idx:
            for j in B_nodes_idx:
                B_adj[i, j] = adj_matrix[i, j]
        
        # 正向路径计算（从node到其他节点）
        dist_out, pred_out = compute_shortest_paths_gpu(adj_matrix, node_idx)
        
        # 计算到B中其他节点的路径，并收集需要添加的节点和边
        nodes_out = set()
        edges_out = set()
        
        # 对于B中每个节点（不包括node自身）
        for target_idx in B_nodes_idx:
            if target_idx != node_idx and dist_out[target_idx] < float('inf'):
                # 回溯路径
                current = target_idx
                while current != node_idx:
                    prev = pred_out[current].item()
                    
                    # 如果节点不在B中，则需要添加
                    if prev not in B_nodes_idx and prev != -1:
                        nodes_out.add(prev)
                    
                    # 如果边不在B中，则需要添加
                    if prev != -1 and B_adj[prev, current] == 0:
                        edges_out.add((idx_to_node[prev], idx_to_node[current]))
                    
                    current = prev
                    if current == -1:  # 防止路径中断
                        break
        
        # 反向路径计算（从其他节点到node）
        # 创建反向图的邻接矩阵
        adj_matrix_reverse = adj_matrix.transpose(0, 1)
        
        dist_in, pred_in = compute_shortest_paths_gpu(adj_matrix_reverse, node_idx)
        
        # 计算从B中其他节点到node的路径，并收集需要添加的节点和边
        nodes_in = set()
        edges_in = set()
        
        # 对于B中每个节点（不包括node自身）
        for source_idx in B_nodes_idx:
            if source_idx != node_idx and dist_in[source_idx] < float('inf'):
                # 回溯路径（注意：这里的路径实际上是反向的，需要调整）
                current = source_idx
                while current != node_idx:
                    prev = pred_in[current].item()
                    
                    # 如果节点不在B中，则需要添加
                    if prev not in B_nodes_idx and prev != -1:
                        nodes_in.add(prev)
                    
                    # 如果边不在B中，则需要添加（注意边的方向是从current到prev）
                    if prev != -1 and B_adj[current, prev] == 0:
                        edges_in.add((idx_to_node[current], idx_to_node[prev]))
                    
                    current = prev
                    if current == -1:  # 防止路径中断
                        break
        
        # 计算补充代价
        cost_out = len(nodes_out) + len(edges_out)
        cost_in = len(nodes_in) + len(edges_in)
        
        # 汇总需要添加的节点和边
        nodes_to_add = nodes_out.union(nodes_in)
        edges_to_add = edges_out.union(edges_in)
        
        # 转换节点索引为原始节点ID
        nodes_to_add_ids = {idx_to_node[idx] for idx in nodes_to_add}
        
        return cost_out, cost_in, nodes_to_add_ids, edges_to_add
    
    except Exception as e:
        print(f"计算补充代价时出错: {e}")
        traceback.print_exc()
        return float('inf'), float('inf'), set(), set()

# 批量计算补充代价（优化GPU利用率）
def batch_calculate_complement_costs_optimized(G, B, task_manager=None):
    """
    批量计算所有候选节点的补充代价，利用GPU并行计算
    
    参数:
    G: 原始网络
    B: 子网络
    task_manager: GPU任务管理器
    
    返回:
    最小代价的节点及其补充信息
    """
    try:
        global use_gpu
        
        # 获取子网络中的节点
        B_nodes = set(B.nodes())
        
        # 如果B中没有节点，直接返回
        if len(B_nodes) == 0:
            return None, float('inf'), set(), set()
        
        # 准备节点索引映射
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        idx_to_node = {i: node for node, i in node_to_idx.items()}
        
        # 获取B中节点的索引
        B_nodes_idx = {node_to_idx[node] for node in B_nodes}
        
        # 计算邻接矩阵
        print("计算邻接矩阵...")
        adj_matrix, success = compute_adjacency_matrix_gpu(G)
        
        if not success:
            print("计算邻接矩阵失败，回退到CPU版本")
            use_gpu = False
            
            # 使用CPU版本的计算
            min_cost = float('inf')
            best_node = None
            best_nodes_to_add = set()
            best_edges_to_add = set()
            
            for node in tqdm(B.nodes(), desc="计算补充代价（CPU）"):
                cost_out, cost_in, nodes_to_add, edges_to_add = calculate_complement_cost(G, B, node)
                total_cost = cost_out + cost_in
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_node = node
                    best_nodes_to_add = nodes_to_add
                    best_edges_to_add = edges_to_add
            
            return best_node, min_cost, best_nodes_to_add, best_edges_to_add
        
        # 确定批处理大小
        # 对于大图，使用较小的批处理大小以减少内存使用
        if len(G) > 10000:
            batch_size = 4
        elif len(G) > 5000:
            batch_size = 8
        else:
            batch_size = 16
            
        # 如果节点数太多，减小批处理大小
        if len(B_nodes) > 500:
            batch_size = max(2, batch_size // 4)
            print(f"子网络节点较多 ({len(B_nodes)}), 减小批处理大小为 {batch_size}")
            
        # GPU版本不使用多个CUDA流，而是顺序处理以避免资源竞争
        if use_gpu:
            print(f"使用GPU计算补充代价，批处理大小: {batch_size}")
        else:
            print(f"使用CPU计算补充代价")
        
        # 初始化结果变量
        min_cost = float('inf')
        best_node = None
        best_nodes_to_add = set()
        best_edges_to_add = set()
        
        # 将节点分成批次
        B_nodes_list = list(B.nodes())
        # 通过节点度排序，先处理度高的节点（可能是更好的候选节点）
        B_nodes_list.sort(key=lambda x: G.degree(x), reverse=True)
        
        # 主循环：遍历所有节点批次
        for batch_start in tqdm(range(0, len(B_nodes_list), batch_size), desc="计算补充代价"):
            # 检查内存使用情况
            if is_memory_critical():
                print("内存使用过高，清理缓存...")
                if use_gpu:
                    torch.cuda.empty_cache()
                # 如果仍然内存过高，可能需要进一步减少批处理大小
                batch_size = max(1, batch_size // 2)
                print(f"减小批处理大小为 {batch_size}")
            
            # 当前批次的节点
            current_batch = B_nodes_list[batch_start:batch_start + batch_size]
            batch_results = []
            
            # 对当前批次的节点计算
            for node in current_batch:
                node_idx = node_to_idx[node]
                
                if use_gpu:
                    # GPU版本
                    cost_out, cost_in, nodes_to_add, edges_to_add = calculate_complement_cost_gpu(
                        adj_matrix, B_nodes_idx, node_idx, node_to_idx, idx_to_node
                    )
                else:
                    # CPU版本
                    cost_out, cost_in, nodes_to_add, edges_to_add = calculate_complement_cost(G, B, node)
                
                total_cost = cost_out + cost_in
                batch_results.append((node, total_cost, nodes_to_add, edges_to_add))
                
                # 清理内存
                if use_gpu and batch_size > 1:
                    torch.cuda.empty_cache()
            
            # 处理批次结果
            for node, total_cost, nodes_to_add, edges_to_add in batch_results:
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_node = node
                    best_nodes_to_add = nodes_to_add
                    best_edges_to_add = edges_to_add
            
            # 提前终止：如果我们找到了一个相当好的解决方案，可以提前终止
            if min_cost < 10 and batch_start >= len(B_nodes_list) // 4:
                print(f"找到好的解决方案 (代价 = {min_cost})，提前终止优化")
                break
        
        print(f"最佳中心节点: {best_node}, 代价: {min_cost}, 添加节点: {len(best_nodes_to_add)}, 添加边: {len(best_edges_to_add)}")
        return best_node, min_cost, best_nodes_to_add, best_edges_to_add
        
    except Exception as e:
        print(f"批量计算补充代价时出错: {e}")
        traceback.print_exc()
        
        if use_gpu:
            print("GPU计算出错，切换到CPU模式...")
            use_gpu = False  # 出错时切换到CPU模式
        
        # 使用CPU版本的计算
        min_cost = float('inf')
        best_node = None
        best_nodes_to_add = set()
        best_edges_to_add = set()
        
        try:
            for node in tqdm(B.nodes(), desc="计算补充代价（CPU回退）"):
                cost_out, cost_in, nodes_to_add, edges_to_add = calculate_complement_cost(G, B, node)
                total_cost = cost_out + cost_in
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_node = node
                    best_nodes_to_add = nodes_to_add
                    best_edges_to_add = edges_to_add
                    
            return best_node, min_cost, best_nodes_to_add, best_edges_to_add
        except Exception as e2:
            print(f"CPU版计算也出错: {e2}")
            # 返回空结果，表示无法计算
            return None, float('inf'), set(), set()

# CPU版本的补充代价计算函数（保留以便在GPU版本失败时回退）
def calculate_complement_cost(G, B, node):
    """
    计算将节点node作为中心节点的补充代价
    
    参数:
    G: 原始网络
    B: 子网络
    node: 候选中心节点
    
    返回:
    cost_out: 正向路径补充代价
    cost_in: 反向路径补充代价
    nodes_to_add: 需要添加的节点
    edges_to_add: 需要添加的边
    """
    # 获取子网络中的节点
    B_nodes = set(B.nodes())
    
    # 正向路径计算（从node到其他节点）
    paths_out = {}
    nodes_out = set()
    edges_out = set()
    
    # 使用BFS计算最短路径
    shortest_paths = nx.single_source_shortest_path(G, node)
    
    # 计算到子网络中其他节点的路径
    for target in B_nodes:
        if target != node and target in shortest_paths:
            path = shortest_paths[target]
            paths_out[target] = path
            
            # 记录路径中不在B中的节点和边
            for i in range(len(path)):
                if path[i] not in B_nodes:
                    nodes_out.add(path[i])
                
                if i > 0 and (path[i-1], path[i]) not in B.edges():
                    edges_out.add((path[i-1], path[i]))
    
    # 反向路径计算（从其他节点到node）
    paths_in = {}
    nodes_in = set()
    edges_in = set()
    
    # 创建反向图
    G_reverse = G.reverse()
    
    # 使用BFS计算最短路径
    shortest_paths_reverse = nx.single_source_shortest_path(G_reverse, node)
    
    # 计算从子网络中其他节点到node的路径
    for source in B_nodes:
        if source != node and source in shortest_paths_reverse:
            path = shortest_paths_reverse[source]
            
            # 反向路径需要反转回来
            path = path[::-1]
            paths_in[source] = path
            
            # 记录路径中不在B中的节点和边
            for i in range(len(path)):
                if path[i] not in B_nodes:
                    nodes_in.add(path[i])
                
                if i > 0 and (path[i-1], path[i]) not in B.edges():
                    edges_in.add((path[i-1], path[i]))
    
    # 计算补充代价
    cost_out = len(nodes_out) + len(edges_out)
    cost_in = len(nodes_in) + len(edges_in)
    
    # 汇总需要添加的节点和边
    nodes_to_add = nodes_out.union(nodes_in)
    edges_to_add = edges_out.union(edges_in)
    
    return cost_out, cost_in, nodes_to_add, edges_to_add

# 强联通网络构建函数
def build_strongly_connected_network(G, B):
    """
    将子网络B补充为强联通图
    
    参数:
    G: 原始网络
    B: 子网络
    
    返回:
    补充后的强联通图、添加的节点数、添加的边数、添加的节点ID列表、添加的边列表
    """
    global use_gpu
    
    # 如果B中没有节点或只有一个节点，直接返回
    if len(B) <= 1:
        return B, 0, 0, [], []
    
    # 创建任务管理器但不抛出异常
    task_manager = None
    if use_gpu:
        try:
            task_manager = ContinuousGPUTaskManager(num_streams=4)
            task_manager.start_background_task()
        except Exception as e:
            print(f"初始化GPU任务管理器错误: {e}")
            # 不切换到CPU模式，继续使用GPU进行主要计算
    
    try:
        # 使用批量计算找到最佳中心节点
        best_node, min_cost, best_nodes_to_add, best_edges_to_add = batch_calculate_complement_costs_optimized(G, B, task_manager)
        
        # 创建补充后的网络
        B_complement = B.copy()
        
        # 添加节点
        for node in best_nodes_to_add:
            B_complement.add_node(node)
        
        # 添加边
        for u, v in best_edges_to_add:
            B_complement.add_edge(u, v)
        
        # 停止任务管理器
        if task_manager and task_manager.running:
            task_manager.stop()
            
        return B_complement, len(best_nodes_to_add), len(best_edges_to_add), list(best_nodes_to_add), list(best_edges_to_add)
    
    except Exception as e:
        print(f"构建强联通网络时出错: {e}")
        traceback.print_exc()
        
        # 停止任务管理器
        if task_manager and task_manager.running:
            task_manager.stop()
            
        # 出错时切换到CPU版本
        print("切换到CPU版本进行计算")
        use_gpu = False
        
        # 使用CPU版本的计算
        min_cost = float('inf')
        best_node = None
        best_nodes_to_add = set()
        best_edges_to_add = set()
        
        # 计算每个节点的补充代价
        for node in tqdm(B.nodes(), desc="计算补充代价（CPU）"):
            cost_out, cost_in, nodes_to_add, edges_to_add = calculate_complement_cost(G, B, node)
            total_cost = cost_out + cost_in
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_node = node
                best_nodes_to_add = nodes_to_add
                best_edges_to_add = edges_to_add
        
        # 创建补充后的网络
        B_complement = B.copy()
        
        # 添加节点
        for node in best_nodes_to_add:
            B_complement.add_node(node)
        
        # 添加边
        for u, v in best_edges_to_add:
            B_complement.add_edge(u, v)
        
        return B_complement, len(best_nodes_to_add), len(best_edges_to_add), list(best_nodes_to_add), list(best_edges_to_add)

def get_expression_data_for_mce(G, expression_file, dictionary_file, sample_name, cached_data=None):
    """
    获取网络中每个节点(基因)的表达量
    
    参数:
    G -- 网络对象
    expression_file -- 基因表达量文件路径
    dictionary_file -- 基因ID映射字典文件路径
    sample_name -- 要获取表达量的样本ID,从文件名中提取
    cached_data -- 可选，包含已加载表达数据的字典，格式: {
        'expr_df': 表达量数据框, 
        'target_column': 目标样本列名,
        'ncbi_to_ensembl': ID映射字典
    }
    
    返回:
    节点表达量字典 {node_id: expression_value}
    """
    try:
        # 获取网络中的节点ID（NCBI ID）
        node_ids = list(G.nodes())
        
        # 检查是否提供了缓存数据
        if cached_data and 'ncbi_to_ensembl' in cached_data and 'expr_df' in cached_data and 'target_column' in cached_data:
            print("使用缓存的表达量数据...")
            ncbi_to_ensembl = cached_data['ncbi_to_ensembl']
            expr_df = cached_data['expr_df']
            target_column = cached_data['target_column']
        else:
            print("加载基因表达量数据...")
            
            # 加载基因ID映射字典
            dict_df = pd.read_csv(dictionary_file, encoding='latin1')
            dict_df['ncbi_gene_id'] = dict_df['ncbi_gene_id'].astype(str)
            
            # 创建NCBI ID到Ensembl ID的映射
            ncbi_to_ensembl = {}
            for idx, row in dict_df.iterrows():
                if pd.notna(row['ncbi_gene_id']) and pd.notna(row['ensembl_gene_id']):
                    ncbi_to_ensembl[row['ncbi_gene_id']] = row['ensembl_gene_id']
            
            # 加载表达量数据(所有列)
            print(f"从文件加载表达量数据: {expression_file}")
            expr_df = pd.read_csv(expression_file, sep='\t')
            
            if sample_name is None:
                print("警告: 未提供样本ID，将使用第二列作为通用表达量")
                # 如果没有样本ID，使用第二列（假设是通用表达量）
                expr_df.columns = ['Ensembl_ID'] + [f'Sample_{i}' for i in range(1, len(expr_df.columns))]
                target_column = expr_df.columns[1]
            else:
                print(f"为样本 {sample_name} 查找对应的表达列...")
                # 查找包含样本ID的列
                target_column = None
                for col in expr_df.columns:
                    if sample_name in col:
                        target_column = col
                        print(f"找到匹配的表达列: {target_column}")
                        break
                
                # 如果仍然找不到匹配列，使用第二列作为备选
                if target_column is None:
                    print(f"警告: 未找到样本 {sample_name} 对应的表达列，将使用第二列作为通用表达量")
                    if len(expr_df.columns) > 1:
                        target_column = expr_df.columns[1]
                    else:
                        print("错误: 表达文件格式不正确")
                        return {}
            
            # 设置Ensembl_ID为索引
            expr_df.set_index(expr_df.columns[0], inplace=True)
        
        # 为每个节点获取表达量
        nodes_expression = {}
        missing_expr_count = 0
        zero_expr_count = 0
        
        print(f"为 {len(node_ids)} 个节点获取表达量数据...")
        for node_id in node_ids:
            if node_id in ncbi_to_ensembl:
                ensembl_id = ncbi_to_ensembl[node_id]
                if ensembl_id in expr_df.index:
                    expression = float(expr_df.loc[ensembl_id, target_column])
                    expression = (2 ** expression) - 1
                    nodes_expression[node_id] = expression
                    if expression == 0:
                        zero_expr_count += 1
                else:
                    nodes_expression[node_id] = 0.0
                    missing_expr_count += 1
            else:
                nodes_expression[node_id] = 0.0
                missing_expr_count += 1
        
        print(f"完成表达量数据获取: {len(node_ids) - missing_expr_count} 个节点有有效表达量，{missing_expr_count} 个节点无法映射到表达量")
        print(f"表达量为0的节点数: {zero_expr_count}")
        
        # 如果这是新加载的数据，返回可缓存的数据结构
        if cached_data is None:
            return nodes_expression, {
                'expr_df': expr_df,
                'target_column': target_column,
                'ncbi_to_ensembl': ncbi_to_ensembl
            }
        else:
            return nodes_expression, cached_data
    
    except Exception as e:
        print(f"获取表达量数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, None


def calculate_normalized_expression(nodes_expression, graph_nodes):
    """
    计算归一化的表达量向量π = x/||x||₁
    
    参数:
    nodes_expression -- 节点表达量字典 {node_id: expression_value}
    graph_nodes -- 图中节点的有序列表，用于确保表达量与邻接矩阵顺序一致
    """
    # 如果提供了图的节点顺序，则按照该顺序获取表达量
    if graph_nodes is not None:
        # 按照图中节点的顺序获取表达量
        node_ids = graph_nodes
        x = np.array([nodes_expression.get(node_id, 0.0) for node_id in node_ids])
        print(f"按照图中的节点顺序获取表达量，共 {len(node_ids)} 个节点")
    else:
        # 否则按照字典的顺序获取
        node_ids = list(nodes_expression.keys())
        x = np.array([nodes_expression[node_id] for node_id in node_ids])
        print("警告：未提供图节点顺序，使用字典默认顺序")
    
    # 计算x的L1范数
    x_norm_l1 = np.sum(np.abs(x))
    
    # 避免除以零
    if x_norm_l1 == 0:
        print("警告: 表达量之和为零，无法计算归一化表达量，返回均匀分布")
        pi = np.ones_like(x) / len(x)
    else:
        pi = x / x_norm_l1
    
    print(f"原始表达量最大值: {np.max(x):.4f}, 最小值: {np.min(x):.10f}, 均值: {np.mean(x):.10f}")

    return node_ids, x, pi


def solve_nonlinear_equations(A, pi, max_iter=10000, tol=1e-6):
    """
    使用迭代方法求解非线性方程组，得到α和β
    使用GPU加速(如果可用)
    """
    print("求解非线性方程组...")
    
    n = len(pi)
    
    # 如果启用了GPU加速
    if HAS_CUPY:
        # 将NumPy数组转换为CuPy数组
        A_gpu = cp.array(A, dtype=cp.float64)
        pi_gpu = cp.array(pi, dtype=cp.float64)

        # 创建B矩阵：B[i,j] = π[i] * A[i,j]
        B_gpu = cp.zeros_like(A_gpu)
        for i in range(n):
            B_gpu[i, :] = pi_gpu[i] * A_gpu[i, :]
        
        # 初始化β为全1向量
        beta_gpu = cp.ones(n)
        
        # 迭代求解
        for iter_count in range(max_iter):
            try:
                # 1. α⁽ᵐ⁾ = 1/(A·β⁽ᵐ⁻¹⁾)
                A_beta = A_gpu.dot(beta_gpu)
                
                alpha_gpu = 1.0 / A_beta
                
                # 2. β⁽ᵐ⁾ = π./(B·α⁽ᵐ⁾)
                B_alpha = B_gpu.transpose().dot(alpha_gpu)
                beta_new_gpu = pi_gpu / B_alpha
               
                # 计算beta的变化，用于判断收敛性
                beta_diff = cp.linalg.norm(beta_new_gpu - beta_gpu) / (cp.linalg.norm(beta_gpu))
                
                # 更新β
                beta_gpu = beta_new_gpu
                
                if beta_diff < tol:
                    break
                    
            except Exception as e:
                print(f"GPU迭代过程中出错: {str(e)}")
                break
        
        # 计算最终的α
        A_beta = A_gpu.dot(beta_gpu)
        
        alpha_gpu = 1.0 / A_beta
        
        # 将结果转回CPU
        alpha = cp.asnumpy(alpha_gpu)
        beta = cp.asnumpy(beta_gpu)
    else:
        # CPU版本
        # 创建B矩阵：B[i,j] = π[i] * A[i,j]
        B = np.zeros_like(A)
        for i in range(n):
            B[i, :] = pi[i] * A[i, :]
        
        # 初始化β为全1向量
        beta = np.ones(n)
        
        # 迭代求解
        for iter_count in range(max_iter):
            try:
                # 1. α⁽ᵐ⁾ = 1/(A·β⁽ᵐ⁻¹⁾)
                A_beta = A.dot(beta)
                alpha = 1.0 / A_beta
                
                # 2. β⁽ᵐ⁾ = π./(B·α⁽ᵐ⁾)
                B_alpha = B.transpose().dot(alpha)
                beta_new = pi / B_alpha
                
                # 计算beta的变化，用于判断收敛性
                beta_diff = np.linalg.norm(beta_new - beta) / (np.linalg.norm(beta))
                
                # 更新β
                beta = beta_new
                
                if beta_diff < tol:
                    break
                    
            except Exception as e:
                print(f"CPU迭代过程中出错: {str(e)}")
                break
        
        # 计算最终的α
        A_beta = A.dot(beta)
        alpha = 1.0 / A_beta
    
    return alpha, beta


def calculate_mce_value(pi, alpha, beta):
    """
    计算MCE值
    """
    # 根据公式计算MCE值: MCE(x) = -∑i πi log2(αi * βi * πi)
    products = alpha * beta * pi
    
    # 使用以2为底的对数
    logs = np.log2(products)
    
    mce_value = -np.sum(pi * logs)
    
    # 如果结果是NaN，返回一个默认值
    if np.isnan(mce_value):
        print("警告: MCE计算结果为NaN，返回默认值0")
        return 0.0
    
    return mce_value


def calculate_max_mce(A):
    """
    计算网络的最大可能MCE值
    """
    # 计算邻接矩阵中非零元素的个数，也就是d = ∑ di
    d = np.count_nonzero(A)
    
    # 最大MCE = log2(d)
    max_mce = np.log2(d)
    
    return max_mce


def normalize_mce(mce, max_mce):
    """
    归一化MCE值到[0,1]区间
    """
    return mce / max_mce


def calculate_mce_for_pathway(G, sample_name, cached_expression_data=None):
    """
    计算单个通路的MCE值
    
    参数:
    G -- 通路网络对象
    sample_name -- 样本名称
    cached_expression_data -- 可选，已缓存的表达数据
    
    返回:
    包含MCE计算结果的字典
    """
    try:
        # 如果节点数量太少，无法计算MCE
        if G.number_of_nodes() <= 1:
            print(f"通路网络节点数量 ({G.number_of_nodes()}) 过少，无法计算MCE值")
            return {
                'mce': None,
                'max_mce': None,
                'normalized_mce': None
            }
            
        print(f"计算通路的MCE值...")
        
        # 获取网络的邻接矩阵和节点顺序
        A = nx.to_numpy_array(G)
        # 确保添加自环 (对角线元素设为1)
        np.fill_diagonal(A, 1)
        
        # 保存图的节点顺序，确保后续计算的一致性
        graph_nodes = list(G.nodes())
        
        print(f"获取到图中节点顺序，共 {len(graph_nodes)} 个节点")
        
        # 获取基因表达量数据
        # 根据样本类型选择表达量文件
        if "LUSC" in sample_name:
            expression_file = "/root/autodl-tmp/MCE/expression/expression/TCGA-LUSC.star_fpkm_processed.tsv"
        else:
            expression_file = "/root/autodl-tmp/MCE/expression/expression/TCGA-LUAD.star_fpkm_processed.tsv"
            
        dictionary_file = "/root/autodl-tmp/MCE/MCE_t/gene_id_dictionary_unique.csv"
        
        # 传递样本名称和缓存数据给get_expression_data函数
        nodes_expression, updated_cache = get_expression_data_for_mce(G, expression_file, dictionary_file, sample_name, cached_expression_data)
        if not nodes_expression:
            print("无法获取表达量数据，无法计算MCE值")
            return {
                'mce': None,
                'max_mce': None,
                'normalized_mce': None
            }
        
        # 计算归一化的表达量向量π，使用图的节点顺序
        node_ids, x, pi = calculate_normalized_expression(nodes_expression, graph_nodes)
        
        # 确认节点顺序一致性
        if node_ids != graph_nodes:
            print("警告：节点顺序不一致！")
            if len(node_ids) == len(graph_nodes):
                print("但节点数量相同，继续计算")
            else:
                print("节点数量不同，计算结果可能不准确")
        
        # 求解非线性方程组，得到α和β
        alpha, beta = solve_nonlinear_equations(A, pi)
        
        # 计算MCE值
        mce_value = calculate_mce_value(pi, alpha, beta)
        
        # 计算最大可能的MCE值
        max_mce = calculate_max_mce(A)
        
        # 计算归一化的MCE值
        normalized_mce = normalize_mce(mce_value, max_mce)
        
        print(f"计算完成: MCE={mce_value:.6f}, 最大MCE={max_mce:.6f}, 归一化MCE={normalized_mce:.6f}")
        
        # 返回结果和更新的缓存
        result = {
            'mce': mce_value,
            'max_mce': max_mce,
            'normalized_mce': normalized_mce
        }
        
        return result, updated_cache
        
    except Exception as e:
        print(f"计算MCE值时出错: {str(e)}")
        traceback.print_exc()
        return {
            'mce': None,
            'max_mce': None,
            'normalized_mce': None
        }, None

def process_pathway_networks(sample_network_file, output_dir, sample_name, pathway_gene_ids_file):
    """
    处理所有通路网络，将它们补充为强联通图
    
    参数:
    sample_network_file: 样本网络文件路径
    output_dir: 输出目录
    sample_name: 样本名称
    pathway_gene_ids_file: 通路基因节点文件路径
    
    返回:
    成功处理的通路数, 失败的通路数
    """
    # 确保输出目录存在
    ensure_directory(output_dir)
    
    # 确保MCE结果目录存在
    mce_output_dir = "/root/autodl-tmp/Pathway_MCE"
    ensure_directory(mce_output_dir)
    
    # 创建MCE结果文件路径
    mce_result_file = os.path.join(mce_output_dir, f"{sample_name}_Pathway_MCE.csv")
    mce_data = []
    
    # 为表达数据创建缓存
    expression_data_cache = None
    
    # 加载样本网络的归一化MCE值
    sample_network_mce = None
    sample_network_mce_file = "/root/autodl-tmp/MCE/MCE_t/results/Hsa_teat.csv"
    if os.path.exists(sample_network_mce_file):
        try:
            sample_mce_df = pd.read_csv(sample_network_mce_file)
            # 查找当前样本的MCE值
            for _, row in sample_mce_df.iterrows():
                if row['sample_id'] == sample_name:
                    sample_network_mce = row['normalized_mce']
                    print(f"找到样本 {sample_name} 的归一化MCE值: {sample_network_mce}")
                    break
            
            if sample_network_mce is None:
                print(f"警告: 在Hsa_teat.csv中未找到样本 {sample_name} 的归一化MCE值")
        except Exception as e:
            print(f"读取样本网络MCE文件出错: {e}")
            sample_network_mce = None
    else:
        print(f"警告: 样本网络MCE文件不存在: {sample_network_mce_file}")
    
    # 如果MCE结果文件已存在，加载已有结果
    if os.path.exists(mce_result_file):
        try:
            mce_df = pd.read_csv(mce_result_file)
            mce_data = mce_df.to_dict('records')
            print(f"已加载 {len(mce_data)} 条已有MCE计算结果")
            processed_mce_pathways = set(mce_df['pathway_id'].tolist()) if 'pathway_id' in mce_df.columns else set()
        except Exception as e:
            print(f"读取MCE结果文件出错: {e}")
            mce_data = []
            processed_mce_pathways = set()
    else:
        processed_mce_pathways = set()
    
    # 原有代码逻辑
    # 检查此样本的统计文件
    stats_file = os.path.join(output_dir, f"{sample_name}_network_stats.csv")
    previously_processed_pathways = set()
    
    # 如果统计文件存在，加载已处理的通路记录
    if os.path.exists(stats_file):
        print(f"发现样本 {sample_name} 的统计文件，将加载已处理记录...")
        try:
            stats_df = pd.read_csv(stats_file)
            # 获取已处理的通路ID
            previously_processed_pathways = set(stats_df['pathway_id'].values) if 'pathway_id' in stats_df.columns else set()
            print(f"已加载 {len(previously_processed_pathways)} 条已处理的通路记录")
        except Exception as e:
            print(f"读取统计文件出错: {e}")
    
    # 检查输出目录中已存在的结果文件
    already_processed_files = set()
    for f in os.listdir(output_dir):
        if f.endswith('.graphml'):
            already_processed_files.add(f.split('.')[0])  # 不包含扩展名
    print(f"输出目录中已有 {len(already_processed_files)} 个处理结果文件")
    
    # 只读取一次样本网络，保存在内存中供所有通路处理使用
    print(f"正在读取样本网络: {sample_network_file}")
    start_time = time.time()
    sample_network = nx.read_graphml(sample_network_file)
    read_time = time.time() - start_time
    print(f"样本网络读取耗时: {read_time:.2f}秒")
    print(f"样本网络包含 {sample_network.number_of_nodes()} 个节点和 {sample_network.number_of_edges()} 条边")
    
    # 样本网络中的节点ID集合（都是字符串类型）
    sample_node_ids = set(sample_network.nodes())
    print(f"样本网络中的节点ID示例: {list(sample_node_ids)[:5]}")
    
    # 读取通路基因ID文件
    print(f"正在读取通路基因ID文件: {pathway_gene_ids_file}")
    try:
        pathway_df = pd.read_csv(pathway_gene_ids_file)
        print(f"通路基因ID文件包含 {len(pathway_df.columns)} 个通路, {len(pathway_df)} 行数据")
        
        # 分析前几个通路的基因ID格式
        for col in list(pathway_df.columns)[:3]:
            gene_examples = pathway_df[col].dropna().head(3).tolist()
            print(f"通路 {col} 的基因ID示例: {gene_examples}")
    except Exception as e:
        print(f"读取通路基因ID文件出错: {e}")
        return 0, 0
    
    # 获取通路列名（即通路ID）
    pathway_columns = list(pathway_df.columns)
    print(f"发现 {len(pathway_columns)} 个通路")
    
    # 初始化记录添加节点和边信息的列表
    stats_data = []
    
    # 加载已有统计数据（如果存在）
    if os.path.exists(stats_file):
        try:
            existing_stats = pd.read_csv(stats_file)
            stats_data = existing_stats.to_dict('records')
            print(f"已加载 {len(stats_data)} 条已有统计记录")
        except Exception as e:
            print(f"加载已有统计数据出错: {e}")
            stats_data = []
    
    # 初始化节点和边的映射字典，用于加速后续处理
    print("预处理样本网络...")
    # 创建节点索引映射，避免重复计算
    node_to_idx = {node: i for i, node in enumerate(sample_network.nodes())}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    # 单线程顺序处理通路，更加稳定，避免竞争条件
    print(f"开始处理样本 {sample_name} 的通路网络...")
    
    # 处理通路计数
    successful_pathways = 0
    failed_pathways = 0
    
    # 如果可用GPU内存较小，可以预先计算并缓存邻接矩阵
    adj_matrix = None
    adj_computed = False
    if use_gpu and sample_network.number_of_nodes() < 15000:  # 只为中等大小的网络预计算
        try:
            print("预计算邻接矩阵...")
            adj_matrix, success = compute_adjacency_matrix_gpu(sample_network)
            adj_computed = success
            if success:
                print("邻接矩阵预计算成功，将用于所有通路处理")
            else:
                print("邻接矩阵预计算失败，将在每个通路处理时计算")
        except Exception as e:
            print(f"预计算邻接矩阵时出错: {e}")
            adj_computed = False
    
    # 按照通路大小排序，先处理小的通路
    pathway_sizes = []
    for pathway_name in pathway_columns:
        # 获取该通路的所有非空基因ID数量
        try:
            gene_count = len(pathway_df[pathway_name].dropna())
            pathway_sizes.append((pathway_name, gene_count))
        except Exception as e:
            print(f"获取通路 {pathway_name} 的大小时出错: {e}")
            pathway_sizes.append((pathway_name, 0))
    
    # 对通路按大小排序
    pathway_sizes.sort(key=lambda x: x[1])
    sorted_pathways = [p[0] for p in pathway_sizes]
    print(f"通路已按大小排序，从小到大")
    
    # 增加一个标志，用于判断是否应该停止处理
    should_stop = False
    
    
    for pathway_name in tqdm(sorted_pathways, desc=f"处理样本 {sample_name} 的通路网络"):
        # 如果需要停止，则跳出循环
        if should_stop:
            break
            
        try:
            # 检查通路是否已处理过
            if pathway_name in previously_processed_pathways or pathway_name in already_processed_files:
                print(f"通路 {pathway_name} 已处理过，跳过强联通图构建...")
                successful_pathways += 1
                
                # 检查是否已经计算过MCE值
                if pathway_name in processed_mce_pathways:
                    print(f"通路 {pathway_name} 已计算过MCE值,跳过...")
                    continue
                
                
                # 加载已处理的通路网络文件，计算MCE值
                pathway_file = os.path.join(output_dir, f"{pathway_name}.graphml")
                if os.path.exists(pathway_file):
                    print(f"加载通路网络文件: {pathway_file}")
                    pathway_network = nx.read_graphml(pathway_file)
                    
                    # 计算MCE值，使用缓存的表达数据
                    mce_result, updated_cache = calculate_mce_for_pathway(pathway_network, sample_name, expression_data_cache)
                    
                    # 更新表达数据缓存
                    if updated_cache is not None:
                        expression_data_cache = updated_cache
                    
                    # 计算通路与样本网络归一化MCE的比值
                    mce_ratio = None
                    if sample_network_mce is not None and mce_result['normalized_mce'] is not None:
                        mce_ratio = mce_result['normalized_mce'] / sample_network_mce
                    
                    # 保存MCE计算结果
                    mce_entry = {
                        'pathway_id': pathway_name,
                        'nodes': pathway_network.number_of_nodes(),
                        'edges': pathway_network.number_of_edges(),
                        'mce': mce_result['mce'],
                        'max_mce': mce_result['max_mce'],
                        'normalized_mce': mce_result['normalized_mce'],
                        'mce_ratio': mce_ratio
                    }
                    mce_data.append(mce_entry)
                    
                    # 增量保存MCE结果
                    mce_df = pd.DataFrame(mce_data)
                    mce_df.to_csv(mce_result_file, index=False)
                    print(f"MCE结果已保存到: {mce_result_file}")
                
                continue
            
            # 获取该通路的所有非空基因ID
            # 分析样本网络中节点ID的格式，以确定正确的转换方法
            genes = []
            try:
                # 获取原始基因ID
                raw_genes = pathway_df[pathway_name].dropna()
                
                # 检查样本网络中节点ID的格式
                node_examples = list(sample_node_ids)[:5]
                print(f"样本网络节点ID示例: {node_examples}")
                
                # 判断样本网络节点ID是否为整数格式（无小数点）
                is_integer_format = all('.' not in str(node) for node in node_examples)
                
                if is_integer_format:
                    # 如果样本网络使用整数格式，则移除小数点部分
                    print(f"样本网络使用整数格式的节点ID,将移除小数点部分")
                    # 先转换为浮点数，再转整数，最后转字符串
                    try:
                        genes = raw_genes.astype(float).astype(int).astype(str).tolist()
                    except:
                        # 如果转换失败，直接转字符串并移除小数点部分
                        genes = raw_genes.astype(str).tolist()
                        genes = [g.split('.')[0] if '.' in g else g for g in genes]
                else:
                    # 如果样本网络使用带小数点的格式，保留原格式
                    print(f"样本网络使用带小数点的节点ID,保留原格式")
                    genes = raw_genes.astype(str).tolist()
                
                # 输出转换后的基因ID示例
                gene_examples = genes[:5] if len(genes) >= 5 else genes
                print(f"转换后的基因ID示例: {gene_examples}")
                
            except Exception as e:
                print(f"转换通路 {pathway_name} 的基因ID时出错: {e}")
                # 最安全的备选方案：直接使用字符串
                genes = pathway_df[pathway_name].dropna().astype(str).tolist()
            
            if len(genes) == 0:
                print(f"通路 {pathway_name} 没有基因节点，跳过")
                continue
            
            print(f"通路 {pathway_name} 包含 {len(genes)} 个基因节点")
            
            # 获取样本网络中存在的基因节点
            existing_genes = set(genes).intersection(sample_node_ids)
            if len(existing_genes) == 0:
                print(f"通路 {pathway_name} 在样本网络中没有匹配的基因节点，跳过")
                continue
            elif len(existing_genes) < len(genes):
                print(f"在样本网络中找到 {len(existing_genes)}/{len(genes)} 个通路基因节点 (部分匹配)")
            else:
                print(f"在样本网络中找到 {len(existing_genes)}/{len(genes)} 个通路基因节点 (完全匹配)")
            
            # 从样本网络中构建子网络 - 即使只有部分匹配也继续处理
            B = sample_network.subgraph(existing_genes).copy()
            
            # 输出子网络信息
            print(f"提取的子网络包含 {B.number_of_nodes()} 个节点和 {B.number_of_edges()} 条边")
            
            # 如果子网络为空或只有一个节点，跳过复杂处理
            if B.number_of_nodes() <= 1:
                stats = {
                    'pathway_id': pathway_name,
                    'original_nodes': B.number_of_nodes(),
                    'original_edges': B.number_of_edges(),
                    'added_nodes': 0,
                    'added_edges': 0,
                    'total_nodes': B.number_of_nodes(),
                    'total_edges': B.number_of_edges(),
                    'added_node_ids': '',
                    'added_edge_ids': '',
                    'processing_time': 0
                }
                # 保存空的或单节点网络文件
                output_path = os.path.join(output_dir, f"{pathway_name}.graphml")
                nx.write_graphml(B, output_path)
                stats_data.append(stats)
                successful_pathways += 1
                
                # 记录MCE值为空
                mce_entry = {
                    'pathway_id': pathway_name,
                    'nodes': B.number_of_nodes(),
                    'edges': B.number_of_edges(),
                    'mce': None,
                    'max_mce': None,
                    'normalized_mce': None,
                    'mce_ratio': None
                }
                mce_data.append(mce_entry)
                continue
            
            # 构建强联通图
            start_time = time.time()
            
            # 使用预先计算的邻接矩阵（如果有）
            if adj_computed and adj_matrix is not None:
                # 使用预计算的邻接矩阵来加速处理
                B_nodes_idx = {node_to_idx[node] for node in B.nodes() if node in node_to_idx}
                
                # 创建任务管理器但不抛出异常
                task_manager = None
                if use_gpu:
                    try:
                        task_manager = ContinuousGPUTaskManager(num_streams=4)
                        task_manager.start_background_task()
                    except Exception as e:
                        print(f"初始化GPU任务管理器错误: {e}")
                
                # 找到最佳中心节点
                min_cost = float('inf')
                best_node = None
                best_nodes_to_add = set()
                best_edges_to_add = set()
                
                # 优先处理度高的节点
                B_nodes_list = list(B.nodes())
                B_nodes_list.sort(key=lambda x: sample_network.degree(x), reverse=True)
                
                batch_size = 4  # 使用较小的批处理大小以减少内存使用
                
                for batch_start in range(0, len(B_nodes_list), batch_size):
                    current_batch = B_nodes_list[batch_start:batch_start + batch_size]
                    batch_results = []
                    
                    for node in current_batch:
                        if node not in node_to_idx:
                            continue
                        node_idx = node_to_idx[node]
                        
                        # GPU版本
                        cost_out, cost_in, nodes_to_add, edges_to_add = calculate_complement_cost_gpu(
                            adj_matrix, B_nodes_idx, node_idx, node_to_idx, idx_to_node
                        )
                        
                        total_cost = cost_out + cost_in
                        batch_results.append((node, total_cost, nodes_to_add, edges_to_add))
                    
                    # 处理批次结果
                    for node, total_cost, nodes_to_add, edges_to_add in batch_results:
                        if total_cost < min_cost:
                            min_cost = total_cost
                            best_node = node
                            best_nodes_to_add = nodes_to_add
                            best_edges_to_add = edges_to_add
                
                # 创建补充后的网络
                B_complement = B.copy()
                
                # 添加节点
                for node in best_nodes_to_add:
                    B_complement.add_node(node)
                
                # 添加边
                for u, v in best_edges_to_add:
                    B_complement.add_edge(u, v)
                
                # 停止任务管理器
                if task_manager and task_manager.running:
                    task_manager.stop()
                
                B_complement, added_nodes, added_edges = B_complement, len(best_nodes_to_add), len(best_edges_to_add)
                added_node_ids, added_edge_ids = list(best_nodes_to_add), list(best_edges_to_add)
            else:
                # 使用标准处理流程
                B_complement, added_nodes, added_edges, added_node_ids, added_edge_ids = build_strongly_connected_network(sample_network, B)
            
            end_time = time.time()
            
            # 记录统计信息
            stats = {
                'pathway_id': pathway_name,
                'original_nodes': B.number_of_nodes(),
                'original_edges': B.number_of_edges(),
                'added_nodes': added_nodes,
                'added_edges': added_edges,
                'total_nodes': B_complement.number_of_nodes(),
                'total_edges': B_complement.number_of_edges(),
                'added_node_ids': ','.join(map(str, added_node_ids)),
                'added_edge_ids': ','.join([f"{u}->{v}" for u, v in added_edge_ids]),
                'processing_time': end_time - start_time
            }
            
            # 保存补充后的网络
            output_path = os.path.join(output_dir, f"{pathway_name}.graphml")
            nx.write_graphml(B_complement, output_path)
            
            stats_data.append(stats)
            successful_pathways += 1
            
            # 增量保存统计信息，避免程序中断导致数据丢失
            if successful_pathways % 10 == 0:
                # 保存统计信息
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_csv(stats_file, index=False)
                print(f"增量保存统计信息，当前已成功处理 {successful_pathways} 个通路")
            
            # 每处理5个通路清理一次内存
            if use_gpu and successful_pathways % 5 == 0:
                torch.cuda.empty_cache()
                print(f"已处理 {successful_pathways} 个通路，清理GPU内存")
                
            # 计算MCE值 - 对所有样本计算MCE值
            print(f"\n计算通路 {pathway_name} 的MCE值...")
            
            # 计算MCE值，使用缓存的表达数据
            mce_result, updated_cache = calculate_mce_for_pathway(B_complement, sample_name, expression_data_cache)
            
            # 更新表达数据缓存
            if updated_cache is not None:
                expression_data_cache = updated_cache
                print("表达数据缓存已更新")
            
            # 计算通路与样本网络归一化MCE的比值
            mce_ratio = None
            if sample_network_mce is not None and mce_result['normalized_mce'] is not None:
                mce_ratio = mce_result['normalized_mce'] / sample_network_mce
                print(f"通路与样本网络归一化MCE比值: {mce_ratio:.6f}")
            
            # 保存MCE计算结果
            mce_entry = {
                'pathway_id': pathway_name,
                'nodes': B_complement.number_of_nodes(),
                'edges': B_complement.number_of_edges(),
                'mce': mce_result['mce'],
                'max_mce': mce_result['max_mce'],
                'normalized_mce': mce_result['normalized_mce'],
                'mce_ratio': mce_ratio
            }
            mce_data.append(mce_entry)
            
            # 增量保存MCE结果
            mce_df = pd.DataFrame(mce_data)
            mce_df.to_csv(mce_result_file, index=False)
            print(f"MCE结果已保存到: {mce_result_file}")
                
        except Exception as e:
            print(f"处理通路 {pathway_name} 时出错: {e}")
            traceback.print_exc()
            failed_pathways += 1
            
            # 如果连续失败过多，可能有系统性问题，终止处理
            if failed_pathways > 10 and successful_pathways == 0:
                print("连续失败过多，终止处理")
                should_stop = True
    
    # 清理表达数据缓存
    expression_data_cache = None
    
    # 如果没有成功处理任何通路网络，发出警告
    if not stats_data:
        print(f"警告: 样本 {sample_name} 没有成功处理任何通路网络")
    else:
        # 保存统计信息
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(stats_file, index=False)
        print(f"统计信息已保存到: {stats_file}")
        print(f"样本 {sample_name} 处理完成: 成功 {successful_pathways} 个通路, 失败 {failed_pathways} 个通路")
    
    # 保存最终的MCE结果
    if mce_data:
        mce_df = pd.DataFrame(mce_data)
        mce_df.to_csv(mce_result_file, index=False)
        print(f"MCE结果已保存到: {mce_result_file}")
        print(f"共计算了 {len(mce_data)} 个通路的MCE值")
    
    return successful_pathways, failed_pathways

def main():
    # 设置内存限制，避免完全占用系统内存
    try:
        # 由于我们现在预计算和缓存更多数据，增加内存限制到50GB
        resource.setrlimit(resource.RLIMIT_AS, (50 * 1024 * 1024 * 1024, -1))
        print("设置内存限制为50GB")
    except Exception as e:
        print(f"无法设置内存限制: {e}")
    
    # 设置超时处理（8小时后强制退出）
    def timeout_handler(signum, frame):
        print("程序运行超过8小时，强制退出")
        sys.exit(1)
    
    # 注册SIGALRM信号处理程序
    signal.signal(signal.SIGALRM, timeout_handler)
    # 设置8小时的闹钟
    signal.alarm(28800000)  # 28800秒 = 8小时
    
    # 如果使用GPU，预热它
    if use_gpu:
        print("预热GPU...")
        dummy = torch.ones(1000, 1000, device='cuda')
        dummy = torch.matmul(dummy, dummy)
        del dummy
        torch.cuda.empty_cache()
        
        # 输出GPU信息
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"CUDA版本: {torch.version.cuda}")
    
    # 检查必要的目录和文件
    luad_dir = "/root/autodl-tmp/MCE/LUAD"
    results_base_dir = "/root/autodl-tmp/Final_pathway"
    pathway_gene_ids_file = "/root/autodl-tmp/pathway_gene_ids.csv"
    
    # 确保基础结果目录存在
    ensure_directory(results_base_dir)
    
    # 创建全局进度跟踪文件
    progress_file = os.path.join(results_base_dir, "processing_progress.csv")
    
    # 检查通路基因ID文件是否存在
    if not os.path.exists(pathway_gene_ids_file):
        print(f"错误: 通路基因ID文件不存在: {pathway_gene_ids_file}")
        sys.exit(1)
    
    # 如果进度文件不存在，创建并添加标题行
    if not os.path.exists(progress_file):
        with open(progress_file, 'w') as f:
            f.write("sample_name,status,successful_pathways,failed_pathways,error_message\n")
    
    # 获取所有样本网络文件列表
    if not os.path.exists(luad_dir):
        print(f"错误: 主样本目录不存在: {luad_dir}")
        sys.exit(1)
    
    all_sample_files = [f for f in os.listdir(luad_dir) if f.endswith('.graphml')]
    print(f"在 {luad_dir} 中找到 {len(all_sample_files)} 个样本网络文件")
    
    # 读取全局进度文件，获取每个样本的处理状态
    sample_status = {}
    if os.path.exists(progress_file):
        try:
            progress_df = pd.read_csv(progress_file)
            for _, row in progress_df.iterrows():
                sample_status[row['sample_name']] = row['status']
            print(f"进度文件中包含 {len(sample_status)} 个样本的处理记录")
            completed_samples = sum(1 for status in sample_status.values() if status == 'completed')
            print(f"其中 {completed_samples} 个样本已处理完成")
        except Exception as e:
            print(f"读取进度文件出错: {e}")
    
    # 初始化进度跟踪
    total_samples = len(all_sample_files)
    processed_samples = 0
    successful_samples = 0
    failed_samples = 0
    skipped_samples = 0
    
    # 处理每个样本文件
    for sample_file in all_sample_files:
        sample_name = os.path.splitext(sample_file)[0]
        
        print(f"\n检查样本 {sample_name} ({processed_samples + 1}/{total_samples})")
        
        # 检查这个特定样本是否已经处理过并标记为完成
        if sample_name in sample_status and sample_status[sample_name] == 'completed':
            print(f"样本 {sample_name} 已处理完成，跳过...")
            processed_samples += 1
            successful_samples += 1
            skipped_samples += 1
            continue
        
        # 检查样本网络文件是否存在
        sample_network_file = os.path.join(luad_dir, sample_file)
        if not os.path.exists(sample_network_file):
            print(f"错误: 样本网络文件不存在: {sample_network_file}, 跳过此样本")
            # 记录处理状态
            with open(progress_file, 'a') as f:
                f.write(f"{sample_name},error,sample network file not found,0,0,\n")
            processed_samples += 1
            failed_samples += 1
            continue
        
        # 确保输出目录存在 - 为每个样本创建一个输出目录
        output_dir = os.path.join(results_base_dir, sample_name)
        ensure_directory(output_dir)
        
        print(f"\n开始处理样本 {sample_name} ({processed_samples + 1}/{total_samples})")
        print(f"网络文件: {sample_network_file}")
        print(f"输出目录: {output_dir}")
        print(f"通路基因ID文件: {pathway_gene_ids_file}")
        
        # 处理通路网络
        try:
            successful_pathways, failed_pathways = process_pathway_networks(
                sample_network_file, output_dir, sample_name, pathway_gene_ids_file
            )
            
            # 记录处理状态
            status = "completed" if successful_pathways > 0 else "failed"
            
            # 检查是否已经有记录，如果有则更新，否则追加
            if sample_name in sample_status:
                # 读取现有记录并更新
                progress_df = pd.read_csv(progress_file)
                progress_df.loc[progress_df['sample_name'] == sample_name, 'status'] = status
                progress_df.loc[progress_df['sample_name'] == sample_name, 'successful_pathways'] = successful_pathways
                progress_df.loc[progress_df['sample_name'] == sample_name, 'failed_pathways'] = failed_pathways
                progress_df.to_csv(progress_file, index=False)
            else:
                # 添加新记录
                with open(progress_file, 'a') as f:
                    f.write(f"{sample_name},{status},{successful_pathways},{failed_pathways},,\n")
            
            # 更新内存中的状态记录
            sample_status[sample_name] = status
            
            successful_samples += 1 if status == "completed" else 0
            failed_samples += 1 if status == "failed" else 0
        except Exception as e:
            print(f"处理样本 {sample_name} 时出错: {e}")
            traceback.print_exc()
            
            # 记录处理状态
            error_msg = str(e).replace(',', ';')
            
            # 检查是否已经有记录，如果有则更新，否则追加
            if sample_name in sample_status:
                # 读取现有记录并更新
                progress_df = pd.read_csv(progress_file)
                progress_df.loc[progress_df['sample_name'] == sample_name, 'status'] = "error"
                progress_df.loc[progress_df['sample_name'] == sample_name, 'error_message'] = error_msg
                progress_df.to_csv(progress_file, index=False)
            else:
                # 添加新记录
                with open(progress_file, 'a') as f:
                    f.write(f"{sample_name},error,0,0,{error_msg}\n")
            
            # 更新内存中的状态记录
            sample_status[sample_name] = "error"
            
            failed_samples += 1
        
        processed_samples += 1
        
        # 清理缓存，避免内存泄漏
        if use_gpu:
            torch.cuda.empty_cache()
        
        print(f"已处理 {processed_samples}/{total_samples} 个样本，成功: {successful_samples}，失败: {failed_samples}，跳过: {skipped_samples}")
    
    print(f"\n所有样本处理完成！成功: {successful_samples}/{total_samples}，失败: {failed_samples}/{total_samples}，跳过: {skipped_samples}/{total_samples}")

if __name__ == "__main__":
    main() 