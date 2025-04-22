# step4.3.3_读取tif写入到csv中
import pandas as pd
import rasterio
import os
import numpy as np # 导入 numpy 用于处理 NaN 和 nodata
from tqdm import tqdm
from collections import defaultdict # 用于更方便地组织 TIF 文件

# 1. 定义路径
base_csv_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY'
base_output_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif'
root_dir = r'E:\地理所\论文\中国XCO2论文_2025.04\数据' # 使用原始字符串或正斜杠

# 确保输出目录存在
os.makedirs(base_output_path, exist_ok=True)

# 2. 定义栅格文件根目录和类型映射
type_to_folder = {
    'Lantitude': '纬度栅格',
    'Longtitude': '经度栅格',
    'UnixTime': '每月时间戳的栅格数据',
    'aspect': '坡向数据',
    'slope':'坡度数据',
    'DEM':'DEM',
    'VIIRS':'夜光遥感',
    'ERA5Land':'ERA5',
    'AOD':'气溶胶厚度',
    'CT2019B':'carbon_tracer',
    'landscan':'landscan',
    'odiac1km':'odiac',
    'humanfootprint':'人类足迹数据',
    'OCO2GEOS':'OCO2_GEOS_XCO2同化数据',
    'CAMStcco2':'CAMS',
    'CLCD':'CLCD',
    'MODISLANDCOVER':'modis_landcover',
    'MOD13A2':'NDVI',
}

annual_types = ['Lantitude','Longtitude','aspect','slope','DEM','landscan','humanfootprint','CLCD', 'MODISLANDCOVER']
monthly_types = ['UnixTime', 'VIIRS', 'ERA5Land', 'AOD', 'CT2019B', 'odiac1km', 'OCO2GEOS', 'CAMStcco2', 'MOD13A2']
types = list(type_to_folder.keys())

# 定义年份和月份范围
year = 2018
months = range(1, 13)

print(f"开始处理 {year} 年的 12 个月份数据...")

# 3. 循环处理每个月份
for month in tqdm(months, desc="处理月份"):
    # 3.1 构建当前月份的文件路径
    csv_filename = f"标准栅格XY_{year}_{month:02d}.csv"
    csv_path = os.path.join(base_csv_path, csv_filename)
    
    output_filename = f"统计_{year}_{month:02d}.csv"
    output_path = os.path.join(base_output_path, output_filename)
    
    print(f"\n处理月份: {month:02d}, 读取: {csv_path}")
    
    # 检查输入文件是否存在
    if not os.path.exists(csv_path):
        print(f"警告: 输入文件 {csv_path} 不存在，跳过此月份")
        continue
    
    # 3.2 加载 CSV 表格
    df = pd.read_csv(csv_path)
    
    # 确保 X 和 Y 是整数类型，用于索引
    df['X'] = df['X'].astype(int)
    df['Y'] = df['Y'].astype(int)
    
    # 3.3 预先创建结果列，并用 NaN 填充
    results = defaultdict(lambda: np.full(len(df), np.nan)) # 使用 defaultdict 初始化
    
    # 3.4 按类型处理 TIF 文件
    print(f"开始处理 {month:02d} 月份的栅格数据...")
    for type_name in tqdm(types, desc=f"处理数据类型 (月份 {month:02d})", leave=False):
        folder_name = type_to_folder[type_name]
        base_folder_path = os.path.join(root_dir, folder_name)
        
        if type_name in annual_types:
            # 按年份分组处理
            for data_year, group in tqdm(df.groupby('year'), desc=f"处理 {type_name} (按年)", leave=False):
                tif_filename = f"{type_name}_{data_year}.tif"
                tif_path = os.path.join(base_folder_path, tif_filename)
                
                if not os.path.exists(tif_path):
                    continue # 跳过这个年份对应的所有行
                
                try:
                    with rasterio.open(tif_path) as src:
                        num_bands = src.count
                        nodatas = src.nodatavals # 获取所有波段的 NoData 值
                        
                        # 读取所有波段数据到内存 (如果内存允许)
                        all_bands_data = src.read() # shape is (num_bands, height, width)
                        height, width = src.height, src.width
                        
                        # 遍历该 TIF 文件对应的所有行 (在当前年份分组内)
                        for index in group.index:
                            X = df.loc[index, 'X']
                            Y = df.loc[index, 'Y']
                            
                            # 检查坐标是否在栅格范围内
                            if not (0 <= Y < height and 0 <= X < width):
                                continue # 跳过这个点
                                
                            # 提取每个波段的值
                            for band_idx in range(num_bands): # 波段索引从 0 开始
                                pixel_value = all_bands_data[band_idx, Y, X]
                                nodata_val = nodatas[band_idx]
                                
                                col_name = f"{type_name}_band{band_idx + 1}" # 列名波段从 1 开始
                                # 检查是否为 NoData 值、-9999 或 -9999.0，并替换为 0
                                if (nodata_val is not None and np.isclose(pixel_value.item(), nodata_val)) or \
                                   np.isnan(pixel_value) or \
                                   np.isclose(pixel_value.item(), -9999) or \
                                   np.isclose(pixel_value.item(), -9999.0):
                                    results[col_name][index] = 0 # 使用 0 替代 NaN 表示 NoData
                                else:
                                    results[col_name][index] = pixel_value
                                    
                except rasterio.RasterioIOError as e:
                    print(f"错误：无法打开或读取文件 {tif_path}: {e}")
                except Exception as e:
                    print(f"处理文件 {tif_path} 时发生未知错误: {e}")
                    
        elif type_name in monthly_types:
            # 按年份和月份分组处理
            for (data_year, data_month), group in tqdm(df.groupby(['year', 'month']), desc=f"处理 {type_name} (按年月)", leave=False):
                tif_filename = f"{type_name}_{data_year}_{data_month:02d}.tif"
                tif_path = os.path.join(base_folder_path, tif_filename)
                
                if not os.path.exists(tif_path):
                    continue # 跳过这个年月对应的所有行
                    
                try:
                    with rasterio.open(tif_path) as src:
                        num_bands = src.count
                        nodatas = src.nodatavals
                        all_bands_data = src.read()
                        height, width = src.height, src.width
                        
                        # 遍历该 TIF 文件对应的所有行 (在当前年月分组内)
                        for index in group.index:
                            X = df.loc[index, 'X']
                            Y = df.loc[index, 'Y']
                            
                            if not (0 <= Y < height and 0 <= X < width):
                                continue
                                
                            for band_idx in range(num_bands):
                                pixel_value = all_bands_data[band_idx, Y, X]
                                nodata_val = nodatas[band_idx]
                                col_name = f"{type_name}_band{band_idx + 1}"
                                
                                # 检查是否为 NoData 值、-9999 或 -9999.0，并替换为 0
                                if (nodata_val is not None and np.isclose(pixel_value.item(), nodata_val)) or \
                                   np.isnan(pixel_value) or \
                                   np.isclose(pixel_value.item(), -9999) or \
                                   np.isclose(pixel_value.item(), -9999.0):
                                    results[col_name][index] = 0
                                else:
                                    results[col_name][index] = pixel_value
                                    
                except rasterio.RasterioIOError as e:
                    print(f"错误：无法打开或读取文件 {tif_path}: {e}")
                except Exception as e:
                    print(f"处理文件 {tif_path} 时发生未知错误: {e}")
    
    # 3.5 将结果合并回原始 DataFrame
    print(f"合并 {month:02d} 月份的结果到 DataFrame...")
    for col_name, values in results.items():
        df[col_name] = values
    
    # 3.6 将任何剩余的 NaN 值替换为 0
    df = df.fillna(0)
    
    # 3.7 保存处理后的表格
    df.to_csv(output_path, index=False, encoding='utf-8-sig') # 添加 encoding 避免中文乱码
    print(f"月份 {month:02d} 处理完成，结果已保存到 {output_path}")

print("所有 12 个月份处理完成！")