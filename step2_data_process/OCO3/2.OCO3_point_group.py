import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager
from tqdm import tqdm

# 加载数据
input_csv_path = r"E:\地理所\工作\徐州卫星同化_2025.2\OCO2卫星数据\OCO2_xco2_filtered_data_2024_筛选.csv"
output_csv_path = r"E:\地理所\工作\徐州卫星同化_2025.2\OCO2卫星数据\OCO2_xco2_filtered_data_2024_筛选_合并.csv"

# 读取CSV文件
data = pd.read_csv(input_csv_path)

# 设置距离阈值 (单位：米)
distance_threshold = 3000  # 3000米内的点合并为一个点

# 地球半径 (单位：米)
EARTH_RADIUS = 6371000

# 限制比较范围的窗口大小
context_window = 100  # 每个点只与上下100个点比较

# 设置进程数
num_processes = 9

# Haversine公式计算两点之间的球面距离（单位：米）
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS * c

# 子任务函数：处理数据子集
def process_subset(args):
    subset, progress_queue = args
    processed = np.zeros(len(subset), dtype=bool)
    merged_data = []

    for i in range(len(subset)):
        if processed[i]:
            continue

        current_point = subset.iloc[i]
        cluster_points = [current_point]

        start_idx = max(0, i - context_window)
        end_idx = min(len(subset), i + context_window + 1)

        for j in range(start_idx, end_idx):
            if processed[j]:
                continue

            other_point = subset.iloc[j]
            distance = haversine(
                current_point["latitude"], current_point["longitude"],
                other_point["latitude"], other_point["longitude"]
            )

            if distance <= distance_threshold:
                cluster_points.append(other_point)
                processed[j] = True

        cluster_df = pd.DataFrame(cluster_points)
        center_latitude = cluster_df["latitude"].mean()
        center_longitude = cluster_df["longitude"].mean()
        center_xco2 = cluster_df["xco2"].mean()
        center_time = cluster_df["time"].iloc[0]

        max_distance = 0
        for p1 in cluster_points:
            for p2 in cluster_points:
                dist = haversine(
                    p1["latitude"], p1["longitude"],
                    p2["latitude"], p2["longitude"]
                )
                max_distance = max(max_distance, dist)

        merged_data.append({
            "latitude": center_latitude,
            "longitude": center_longitude,
            "xco2": center_xco2,
            "time": center_time,
            "width": max_distance
        })

        # 更新全局进度条
        progress_queue.put(1)

    return merged_data

# 数据分块函数
def split_data(data, num_chunks):
    chunk_size = len(data) // num_chunks
    return [data.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

# 主程序：多进程处理
if __name__ == "__main__":
    print("Splitting data...")
    subsets = split_data(data, num_processes)

    # 创建 Manager 和队列，用于跨进程通信和更新进度条
    manager = Manager()
    progress_queue = manager.Queue()

    # 初始化全局进度条
    total_points = len(data)
    with tqdm(total=total_points) as pbar:
        with Pool(num_processes) as pool:
            # 将子集和队列作为参数传递给每个子任务
            args_list = [(subset, progress_queue) for subset in subsets]

            # 异步处理子任务，并实时更新主进程的进度条
            results_async = pool.map_async(process_subset, args_list)

            while not results_async.ready():
                while not progress_queue.empty():
                    progress_queue.get()
                    pbar.update(1)

            results_async.wait()

        # 获取所有子任务结果
        results = results_async.get()

    print("Merging results...")
    merged_data_final = [item for sublist in results for item in sublist]

    # 保存结果到CSV文件
    merged_df = pd.DataFrame(merged_data_final)
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Processed data saved to: {output_csv_path}")
