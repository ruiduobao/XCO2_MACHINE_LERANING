# 获取OCO3卫星数据整年观测点的经纬度

import xarray as xr
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

# 设置文件路径
file_path = r"E:\地理所\工作\徐州卫星同化_2025.2\OCO3卫星数据\*.nc4"

# 获取2024年的所有NetCDF文件
files = glob.glob(file_path)

# 初始化变量以存储所有数据
all_latitudes = []
all_longitudes = []
all_xco2 = []
all_times = []

# 遍历每个文件，提取xco2, latitude, longitude, time
print("Processing files...")
for file in tqdm(files):
    try:
        # 打开NetCDF文件
        ds = xr.open_dataset(file)
        
        # 提取xco2, latitude, longitude, time变量
        xco2 = ds['xco2']
        latitude = ds['latitude']
        longitude = ds['longitude']
        time = ds['time']  # 时间属性
        
        # 筛选有效数据 (去除缺失值)
        valid_mask = (xco2 != -999999.0) & (latitude != -999999.0) & (longitude != -999999.0) & (time != -999999.0)
        xco2_valid = xco2.where(valid_mask, drop=True).values
        lat_valid = latitude.where(valid_mask, drop=True).values
        lon_valid = longitude.where(valid_mask, drop=True).values
        time_valid = time.where(valid_mask, drop=True).values
        
        # 将有效数据添加到列表中
        all_xco2.extend(xco2_valid)
        all_latitudes.extend(lat_valid)
        all_longitudes.extend(lon_valid)
        
        # 转换时间为标准格式 (numpy.datetime64 -> datetime)
        formatted_times = pd.to_datetime(time_valid, unit='s', origin='unix')
        all_times.extend(formatted_times)
        
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    finally:
        ds.close()

print("Data extraction completed.")

# 转换为NumPy数组以便保存
all_xco2 = np.array(all_xco2)
all_latitudes = np.array(all_latitudes)
all_longitudes = np.array(all_longitudes)
all_times = np.array(all_times)

# 保存数据到CSV文件
output_csv_path = r"E:\地理所\工作\徐州卫星同化_2025.2\OCO3卫星数据\xco2_data_with_time_2024.csv"
data_dict = {
    "latitude": all_latitudes,
    "longitude": all_longitudes,
    "xco2": all_xco2,
    "time": all_times  # 已格式化的时间
}
df = pd.DataFrame(data_dict)
df.to_csv(output_csv_path, index=False)
print(f"Data saved to CSV: {output_csv_path}")
