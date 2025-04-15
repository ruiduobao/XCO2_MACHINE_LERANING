import pandas as pd

# 读取CSV文件
file_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\卫星数据\各个卫星_2018年_各个月汇总.csv'
df = pd.read_csv(file_path)

# 确保xco2列为数值类型
df['xco2'] = pd.to_numeric(df['xco2'], errors='coerce')

# 检查缺失值，确保必要的列没有缺失
if df[['X', 'Y', 'xco2', 'satellite', 'year', 'month', 'longitude', 'latitude']].isna().any().any():
    print("警告：数据中存在缺失值，已跳过相关行")
    df = df.dropna(subset=['X', 'Y', 'xco2', 'satellite', 'year', 'month', 'longitude', 'latitude'])

# 定义卫星精度（统一转换为小写以匹配数据）
precision = {
    'tansat': 2.11,
    'gosat': 1.5,
    'oco2': 0.5
}

# 计算权重（权重与精度的倒数成正比）
weights = {sat: 1 / prec for sat, prec in precision.items()}

# 将satellite列转换为小写，确保与精度字典匹配
df['satellite'] = df['satellite'].str.lower()

# 分组数据：按year, month, X, Y分组
grouped = df.groupby(['year', 'month', 'X', 'Y'])

# 初始化结果列表
results = []

# 遍历每个网格
for (year, month, X, Y), group in grouped:
    # 计算平均经纬度
    avg_longitude = group['longitude'].mean()
    avg_latitude = group['latitude'].mean()
    
    # 获取该网格内的卫星列表（去重）
    satellites = group['satellite'].unique()
    satellites_str = ', '.join(satellites)  # 用逗号分隔记录卫星名称
    
    # 计算数据点数量
    count = len(group)
    
    if len(satellites) == 1:
        # 只有一个卫星，计算平均值
        avg_xco2 = group['xco2'].mean()
        std_xco2 = group['xco2'].std() if count > 1 else 0  # 单个点标准差为0
        results.append({
            'year': year,
            'month': month,
            'X': X,
            'Y': Y,
            'longitude': avg_longitude,
            'latitude': avg_latitude,
            'satellites': satellites_str,
            'xco2': avg_xco2,
            'std_xco2': std_xco2,
            'count': count
        })
    else:
        # 多个卫星，计算加权平均
        weighted_sum = 0
        total_weight = 0
        for satellite in satellites:
            satellite_data = group[group['satellite'] == satellite]
            avg_xco2_sat = satellite_data['xco2'].mean()
            weight = weights.get(satellite, 0)  # 未定义卫星权重为0
            weighted_sum += avg_xco2_sat * weight
            total_weight += weight
        if total_weight > 0:
            weighted_avg_xco2 = weighted_sum / total_weight
            std_xco2 = group['xco2'].std() if count > 1 else 0
            results.append({
                'year': year,
                'month': month,
                'X': X,
                'Y': Y,
                'longitude': avg_longitude,
                'latitude': avg_latitude,
                'satellites': satellites_str,
                'xco2': weighted_avg_xco2,
                'std_xco2': std_xco2,
                'count': count
            })
        else:
            print(f'网格 ({year}, {month}, {X}, {Y}) 的卫星权重为0，无法计算加权平均')

# 将结果转换为DataFrame
result_df = pd.DataFrame(results)

# 保存结果到新的CSV文件
output_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\卫星数据\网格XCO2加权统计_按年月.csv'
result_df.to_csv(output_path, index=False)
print(f'统计完成，结果已保存到 {output_path}')