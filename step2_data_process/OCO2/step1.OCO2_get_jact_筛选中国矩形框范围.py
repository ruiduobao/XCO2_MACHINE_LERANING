import xarray as xr
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

# 设置文件路径
file_path = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\OCO2\原始数据\*.nc4"

# 获取2024年的所有NetCDF文件
files = glob.glob(file_path)

# 初始化变量以存储筛选后的数据
all_latitudes = []
all_longitudes = []
all_xco2 = []
all_times = []

# 筛选条件：仅保留质量标志为0的数据
quality_flag_threshold = 0

# 定义经纬度范围
lat_min = 15  # 南纬15°
lat_max = 57   # 北纬57°
lon_min = 70  # 西经70°
lon_max = 140  # 东经140°

print("Processing files...")
for file in tqdm(files):
    try:
        # 打开NetCDF文件
        ds = xr.open_dataset(file)
        
        # 提取变量
        xco2 = ds['xco2']
        latitude = ds['latitude']
        longitude = ds['longitude']
        time = ds['time']
        quality_flag = ds['xco2_quality_flag']  # 质量标志

        # 筛选有效数据 (去除缺失值并满足质量条件和经纬度范围)
        valid_mask = (
            (xco2 != -999999.0) &
            (latitude != -999999.0) &
            (longitude != -999999.0) &
            (time != -999999.0) &
            (quality_flag == quality_flag_threshold) &
            (latitude >= lat_min) & (latitude <= lat_max) &
            (longitude >= lon_min) & (longitude <= lon_max)
        )
        
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

print("Data extraction and filtering completed.")

# 转换为NumPy数组以便保存
all_xco2 = np.array(all_xco2)
all_latitudes = np.array(all_latitudes)
all_longitudes = np.array(all_longitudes)
all_times = np.array(all_times)

# 保存数据到CSV文件
output_csv_path = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\OCO2\处理的数据\OCO2_xco2_filtered_data_2018_中国区域_good_quality.csv"
data_dict = {
    "latitude": all_latitudes,
    "longitude": all_longitudes,
    "xco2": all_xco2,
    "time": all_times  # 已格式化的时间
}
df = pd.DataFrame(data_dict)
df.to_csv(output_csv_path, index=False)
print(f"Filtered data saved to CSV: {output_csv_path}")