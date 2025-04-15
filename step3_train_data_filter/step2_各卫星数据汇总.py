import os
import pandas as pd

# 定义文件夹路径和对应的卫星名称
folders = {
    'gosat': r'E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\卫星数据\gosat',
    'OCO2': r'E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\卫星数据\OCO2',
    'TANSAT': r'E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\卫星数据\TANSAT'
}

# 定义需要保留的字段
columns_to_keep = ['longitude','latitude','X', 'Y', 'xco2', 'year', 'month', 'day', 'hour', 'minute', 'second']

# 初始化一个空的 DataFrame 用于汇总
all_data = pd.DataFrame()

# 遍历每个文件夹
for satellite, folder_path in folders.items():
    # 遍历文件夹下的所有 CSV 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                # 读取 CSV 文件
                df = pd.read_csv(file_path)
                # 检查是否包含所有需要的字段
                if all(col in df.columns for col in columns_to_keep):
                    # 如果是 TANSAT，将 xco2 乘以 1,000,000
                    if satellite == 'TANSAT':
                        df['xco2'] = df['xco2'] * 1000000
                    # 提取指定字段
                    df = df[columns_to_keep]
                    # 添加 satellite 字段
                    df['satellite'] = satellite
                    # 合并到汇总数据中
                    all_data = pd.concat([all_data, df], ignore_index=True)
                else:
                    print(f'文件 {filename} 缺少部分字段，已跳过')
            except Exception as e:
                print(f'读取文件 {filename} 时出错：{e}')

# 保存汇总后的数据
output_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\卫星数据\各个卫星_2018年_各个月汇总.csv'
all_data.to_csv(output_path, index=False)
print(f'汇总完成，数据已保存到 {output_path}')