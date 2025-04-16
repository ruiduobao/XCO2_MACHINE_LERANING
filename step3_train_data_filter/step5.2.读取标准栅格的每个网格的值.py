# step4.3.3_读取tif写入到csv中
import pandas as pd
import rasterio
import os
import numpy as np # 导入 numpy 用于处理 NaN 和 nodata
from tqdm import tqdm
from collections import defaultdict # 用于更方便地组织 TIF 文件

# 1. 加载 CSV 表格
csv_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY\标准栅格XY_2018_03.csv' # 使用原始字符串或正斜杠
df = pd.read_csv(csv_path)

# 确保 X 和 Y 是整数类型，用于索引
df['X'] = df['X'].astype(int)
df['Y'] = df['Y'].astype(int)

# 2. 定义栅格文件根目录和类型映射
root_dir = r'E:\地理所\论文\中国XCO2论文_2025.04\数据' # 使用原始字符串或正斜杠

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

# 3. 预先创建结果列，并用 NaN 填充
#   (这一步有助于确保即使某些 TIF 文件缺失，列也存在)
#   需要提前知道每个 TIF 可能有多少个波段。为了简化，我们先假设最多 N 个波段，
#   或者在打开每个 TIF 时动态确定并添加列。
#   更健壮的方法是先扫描 TIF 或在循环中添加列。这里我们先在循环中处理。
#   我们用一个字典来存储需要添加的数据，最后再合并回 DataFrame。
results = defaultdict(lambda: np.full(len(df), np.nan)) # 使用 defaultdict 初始化

# 4. 按类型处理 TIF 文件
print("开始处理栅格数据...")
for type_name in tqdm(types, desc="处理数据类型"):
    folder_name = type_to_folder[type_name]
    base_folder_path = os.path.join(root_dir, folder_name)

    if type_name in annual_types:
        # 按年份分组处理
        # Groupby 'year' and iterate through each year group
        for year, group in tqdm(df.groupby('year'), desc=f"处理 {type_name} (按年)", leave=False):
            tif_filename = f"{type_name}_{year}.tif"
            tif_path = os.path.join(base_folder_path, tif_filename)

            if not os.path.exists(tif_path):
                # print(f"警告：年文件 {tif_path} 不存在，跳过年份 {year}")
                continue # 跳过这个年份对应的所有行

            try:
                with rasterio.open(tif_path) as src:
                    num_bands = src.count
                    nodatas = src.nodatavals # 获取所有波段的 NoData 值

                    # 读取所有波段数据到内存 (如果内存允许)
                    # 注意：如果栅格非常大，这里可能需要分块读取，但对于点提取，一次性读取通常更快
                    all_bands_data = src.read() # shape is (num_bands, height, width)
                    height, width = src.height, src.width

                    # 遍历该 TIF 文件对应的所有行 (在当前年份分组内)
                    for index in group.index:
                        X = df.loc[index, 'X']
                        Y = df.loc[index, 'Y']

                        # 检查坐标是否在栅格范围内
                        if not (0 <= Y < height and 0 <= X < width):
                            # print(f"警告：行索引 {index}, 坐标 (X={X}, Y={Y}) 超出 {tif_path} 范围")
                            continue # 跳过这个点

                        # 提取每个波段的值
                        for band_idx in range(num_bands): # 波段索引从 0 开始
                            pixel_value = all_bands_data[band_idx, Y, X]
                            nodata_val = nodatas[band_idx]

                            # 检查是否为 NoData 值
                            # 需要处理 nodata_val 可能为 None 的情况，以及浮点数比较
                            col_name = f"{type_name}_band{band_idx + 1}" # 列名波段从 1 开始
                            if nodata_val is not None and np.isclose(pixel_value.item(), nodata_val):
                                results[col_name][index] = np.nan # 使用 NaN 表示 NoData
                            else:
                                results[col_name][index] = pixel_value

            except rasterio.RasterioIOError as e:
                print(f"错误：无法打开或读取文件 {tif_path}: {e}")
            except Exception as e:
                print(f"处理文件 {tif_path} 时发生未知错误: {e}")


    elif type_name in monthly_types:
        # 按年份和月份分组处理
        # Groupby 'year' and 'month' and iterate through each group
        for (year, month), group in tqdm(df.groupby(['year', 'month']), desc=f"处理 {type_name} (按年月)", leave=False):
            tif_filename = f"{type_name}_{year}_{month:02d}.tif"
            tif_path = os.path.join(base_folder_path, tif_filename)

            if not os.path.exists(tif_path):
                # print(f"警告：月文件 {tif_path} 不存在，跳过年月 {year}-{month:02d}")
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
                            # print(f"警告：行索引 {index}, 坐标 (X={X}, Y={Y}) 超出 {tif_path} 范围")
                            continue

                        for band_idx in range(num_bands):
                            pixel_value = all_bands_data[band_idx, Y, X]
                            nodata_val = nodatas[band_idx]
                            col_name = f"{type_name}_band{band_idx + 1}"
                            if nodata_val is not None and np.isclose(pixel_value.item(), nodata_val):
                                results[col_name][index] = np.nan
                            else:
                                results[col_name][index] = pixel_value

            except rasterio.RasterioIOError as e:
                print(f"错误：无法打开或读取文件 {tif_path}: {e}")
            except Exception as e:
                print(f"处理文件 {tif_path} 时发生未知错误: {e}")

# 5. 将结果合并回原始 DataFrame
print("合并结果到 DataFrame...")
for col_name, values in results.items():
    df[col_name] = values

# 6. 保存处理后的表格
output_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif\统计_2008_03.csv' # 使用原始字符串或正斜杠
df.to_csv(output_path, index=False, encoding='utf-8-sig') # 添加 encoding 避免中文乱码
print(f"处理完成，结果已保存到 {output_path}")