import os
import h5py
import numpy as np
import pandas as pd

# 文件夹路径
directory = r'E:\地理所\工作\徐州卫星同化_2025.2\gosat\gosat2023'
output_csv = r'E:\地理所\工作\徐州卫星同化_2025.2\gosat\GOSAT2_XCO2_Data.csv'

# 初始化一个空的 DataFrame
all_data = pd.DataFrame()

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    if filename.endswith('.h5'):
        file_path = os.path.join(directory, filename)
        # 打开 HDF5 文件
        with h5py.File(file_path, 'r') as f:
            # 提取 XCO2 数据
            xco2 = f['RetrievalResult_B2_1590/XCO2_B2_1590'][:]
            # 提取经纬度数据
            latitude = f['SoundingGeometry/latitude'][:]
            longitude = f['SoundingGeometry/longitude'][:]
            # 提取观测时间
            observation_time = f['SoundingAttribute/observationTime'][:].astype(str)

        # 过滤无效值 (-999 是无效值)
        valid_mask = (xco2 != -999) & (latitude >= -90) & (latitude <= 90) & (longitude >= -180) & (longitude <= 180)
        xco2_valid = xco2[valid_mask]
        latitude_valid = latitude[valid_mask]
        longitude_valid = longitude[valid_mask]
        observation_time_valid = observation_time[valid_mask]

        # 创建 DataFrame
        data = {
            'Latitude': latitude_valid,
            'Longitude': longitude_valid,
            'XCO2 (ppm)': xco2_valid,
            'Observation Time': observation_time_valid
        }
        df = pd.DataFrame(data)

        # 追加到总的 DataFrame
        all_data = pd.concat([all_data, df], ignore_index=True)

# 保存为 CSV 文件
all_data.to_csv(output_csv, index=False)
print(f"数据已成功保存到 {output_csv}")