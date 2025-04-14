# 导入所需的库
import geopandas as gpd  # 用于处理矢量数据 (GeoPackage)
import rasterio         # 用于处理栅格数据 (GeoTIFF)
import rasterio.sample  # 用于从栅格中采样点位值
import os               # 用于操作系统相关功能 (如文件检查)
import numpy as np      # 用于数值计算 (特别是处理 NoData 值 nan)
import pandas as pd     # 用于创建和处理数据框 (DataFrame) 并输出 CSV
from pathlib import Path # 用于面向对象的文件路径操作
from tqdm import tqdm    # 可选，用于显示进度条

# --- 配置部分 ---
YEAR = 2018  # 设置要处理的年份
# 设置基础数据目录路径 (请根据你的实际路径修改)
BASE_DATA_DIR = Path(r"E:\地理所\论文\中国XCO2论文_2025.04\数据")
# OCO-2 每月点数据所在的目录
OCO2_POINT_DIR = BASE_DATA_DIR / "OCO2" / "处理的数据" / "每个月的数据"
# 设置输出 CSV 文件的目录
OUTPUT_DIR = BASE_DATA_DIR / "Extracted_Features_CSV" / str(YEAR) # 改用新的输出目录名
# 创建输出目录 (如果不存在)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 定义需要提取值的栅格数据源列表 (与之前相同)
RASTER_SOURCES = [
    {'name': 'LULC', 'path': BASE_DATA_DIR / 'LandCover' / 'MCD12Q1', 'time_type': 'annual', 'bands': [1]}, # 示例: 土地覆盖，年数据，提取第1波段
    {'name': 'ERA5_Temp', 'path': BASE_DATA_DIR / 'Meteorology' / 'ERA5_Land_T2M', 'time_type': 'monthly', 'bands': [1]}, # 示例: ERA5 温度，月数据，提取第1波段
    {'name': 'ERA5_WindU', 'path': BASE_DATA_DIR / 'Meteorology' / 'ERA5_Land_U10', 'time_type': 'monthly', 'bands': [1]}, # 示例: ERA5 U风，月数据，提取第1波段
    {'name': 'MODIS_NDVI', 'path': BASE_DATA_DIR / 'Vegetation' / 'MOD13C2_NDVI', 'time_type': 'monthly', 'bands': [1]}, # 示例: MODIS NDVI，月数据，提取第1波段
    {'name': 'AOD', 'path': BASE_DATA_DIR / 'Aerosol' / 'MOD08M3_AOD', 'time_type': 'monthly', 'bands': [1, 2]}, # 示例: AOD，月数据，提取第1和第2波段
    # --- 在这里添加所有其他的栅格数据源配置 ---
]

# --- 主循环：遍历指定年份的每个月 ---
for month in tqdm(range(1, 13), desc="处理月份"):
    month_str = f"{month:02d}" # 格式化月份字符串
    # 构建当前月份 OCO-2 点数据 GeoPackage 文件的完整路径
    point_gpkg = OCO2_POINT_DIR / f"处理的数据{YEAR % 100}_XCO2_{month_str}.gpkg" # !! 根据你的实际文件名调整 !!

    # 检查点文件是否存在
    if not point_gpkg.is_file():
        print(f"警告: 未找到 {YEAR}-{month_str} 的点文件，跳过此月份。路径: {point_gpkg}")
        continue

    print(f"开始处理 {YEAR}-{month_str}...")
    # 使用 GeoPandas 读取点数据 GeoPackage 文件
    gdf = gpd.read_file(point_gpkg)
    # 存储点数据的原始 CRS (虽然输出 CSV 不需要，但重投影时仍需要检查)
    initial_crs = gdf.crs

    # --- 准备输出数据结构 ---
    # 获取 FID (Feature ID)
    # 检查 GeoDataFrame 是否已有 'fid' 列，否则使用其索引作为 FID
    if 'fid' in gdf.columns:
        fids = gdf['fid'].tolist()
        print("使用 GeoPackage 中的 'fid' 列作为 FID。")
    else:
        fids = gdf.index.tolist()
        print("GeoPackage 中无 'fid' 列，使用 GeoDataFrame 索引作为 FID。")

    # 初始化一个字典来存储提取的数据，首先放入 FID
    # 后续每个栅格源提取的值将作为新列添加到这个字典
    output_data = {'fid': fids}
    # 可选：如果你想在 CSV 中包含原始 GPKG 中的其他属性（除了 xco2 和 geometry），可以在这里添加
    # for col in gdf.columns:
    #     if col not in ['geometry', 'fid', 'xco2']: # 假设你不想重复 xco2
    #         output_data[col] = gdf[col].tolist()

    # --- 内层循环：遍历 RASTER_SOURCES 中定义的每个栅格数据源 ---
    for source in tqdm(RASTER_SOURCES, desc=f"栅格源 {month_str}", leave=False):
        raster_path = source['path']
        source_name = source['name']
        time_type = source['time_type']
        bands_to_extract = source.get('bands', None)

        # --- 查找当前月份对应的栅格文件 --- (逻辑同前)
        target_raster = None
        if time_type == 'monthly':
            pattern = f"*{YEAR}_{month_str}*.tif" # !! 根据你的实际文件名调整 !!
            monthly_files = list(raster_path.glob(pattern))
            if monthly_files:
                target_raster = monthly_files[0]
                if len(monthly_files) > 1: print(f"警告: 为 {source_name} 在 {YEAR}-{month_str} 找到多个月度文件，将使用第一个: {target_raster}")
            # else: # 可选的年度备选逻辑 (同前)
                # ...
        elif time_type == 'annual':
            pattern = f"*{YEAR}*.tif" # !! 根据你的实际文件名调整 !!
            annual_files = list(raster_path.glob(pattern))
            if annual_files:
                target_raster = annual_files[0]
                if len(annual_files) > 1: print(f"警告: 为 {source_name} {YEAR} 找到多个年度文件，将使用第一个: {target_raster}")

        if not target_raster or not target_raster.is_file():
            print(f"警告: 未找到 {source_name} 在 {YEAR}-{month_str} 期间的栅格文件。跳过此数据源。")
            # 为这个跳过的数据源的所有目标列填充 NaN
            num_bands = 1 # 假设至少有一个波段，即使文件找不到
            if bands_to_extract:
                process_bands_indices = [b - 1 for b in bands_to_extract if b > 0] # 假设的波段
            else: # 如果未指定波段，但文件未找到，无法知道波段数，跳过填充或假设单波段
                 process_bands_indices = [0] # 假设单波段

            for band_idx in process_bands_indices:
                 band_num = band_idx + 1
                 # 确定列名
                 # 注意：这里的 num_bands 可能不准确，因为文件未找到
                 col_name = f"{source_name}_b{band_num}" if (bands_to_extract and len(bands_to_extract)>1) or num_bands > 1 else source_name
                 output_data[col_name] = [np.nan] * len(gdf) # 用 NaN 填充整列
            continue # 跳到下一个栅格数据源

        # --- 开始提取栅格值 ---
        try:
            with rasterio.open(target_raster) as src:
                gdf_proj = gdf # 先假设 CRS 相同
                # --- CRS 检查与转换 ---
                if gdf.crs != src.crs:
                    print(f"注意: 正在为 {source_name} 将点数据的 CRS 从 {gdf.crs} 转换为 {src.crs}")
                    try:
                        gdf_proj = gdf.to_crs(src.crs)
                    except Exception as e:
                        print(f"错误: 为 {source_name} 重投影点数据时出错: {e}。跳过此数据源。")
                        # 为这个跳过的数据源的所有目标列填充 NaN
                        num_bands = src.count
                        if bands_to_extract is None: process_bands_indices = range(num_bands)
                        else: process_bands_indices = [b - 1 for b in bands_to_extract if 0 < b <= num_bands]
                        for band_idx in process_bands_indices:
                             band_num = band_idx + 1
                             col_name = f"{source_name}_b{band_num}" if num_bands > 1 or (bands_to_extract and len(bands_to_extract)>1) else source_name
                             output_data[col_name] = [np.nan] * len(gdf)
                        continue # 跳到下一个数据源
                # else: # CRS 相同，gdf_proj = gdf 已在上面设置

                # --- 获取点坐标 ---
                coords = [(p.x, p.y) for p in gdf_proj.geometry]

                # --- 采样栅格值 ---
                sampled_data_generator = src.sample(coords, masked=True)

                # --- 确定要处理的波段 ---
                num_bands = src.count
                if bands_to_extract is None:
                    process_bands_indices = range(num_bands) # 0-based index
                else:
                    process_bands_indices = [b - 1 for b in bands_to_extract if 0 < b <= num_bands]
                    if not process_bands_indices:
                         print(f"警告: 为 {source_name} 指定的波段 {bands_to_extract} 无效或超出范围 ({num_bands} bands)。跳过此数据源。")
                         # 为这个跳过的数据源的所有目标列填充 NaN (如果需要明确列出所有可能的列)
                         # ... (填充NaN逻辑，但可能更简单的是不创建列)
                         continue


                # --- 准备临时存储当次采样结果 ---
                # 使用临时字典，因为需要按列填充 output_data
                temp_results = {}
                for band_idx in process_bands_indices:
                    band_num = band_idx + 1
                    col_name = f"{source_name}_b{band_num}" if num_bands > 1 or (bands_to_extract and len(bands_to_extract)>1) else source_name
                    temp_results[col_name] = [] # 初始化空列表

                # --- 处理采样结果生成器 ---
                for point_values in sampled_data_generator:
                    for band_idx in process_bands_indices:
                        band_num = band_idx + 1
                        col_name = f"{source_name}_b{band_num}" if num_bands > 1 or (bands_to_extract and len(bands_to_extract)>1) else source_name
                        if np.ma.is_masked(point_values[band_idx]):
                            temp_results[col_name].append(np.nan)
                        else:
                            temp_results[col_name].append(point_values[band_idx])

                # --- 将提取的值添加到 output_data 字典 ---
                for col_name, values in temp_results.items():
                    if len(values) == len(gdf):
                        output_data[col_name] = values # 将提取的值列表存入主字典
                    else:
                        print(f"错误: 列 {col_name} 的值的数量 ({len(values)}) 与 GDF 行数 ({len(gdf)}) 不匹配。此列未添加。")

        except rasterio.RasterioIOError as rio_err:
             print(f"错误: 使用 Rasterio 打开或读取栅格文件 {target_raster} 时出错: {rio_err}。为 {source_name} 填充 NaN。")
             # 为这个出错的数据源的所有目标列填充 NaN
             # 尝试获取波段数，如果失败则假设单波段
             try: num_bands = rasterio.open(target_raster).count
             except: num_bands = 1
             if bands_to_extract is None: process_bands_indices = range(num_bands)
             else: process_bands_indices = [b - 1 for b in bands_to_extract if 0 < b <= num_bands]
             for band_idx in process_bands_indices:
                  band_num = band_idx + 1
                  col_name = f"{source_name}_b{band_num}" if num_bands > 1 or (bands_to_extract and len(bands_to_extract)>1) else source_name
                  output_data[col_name] = [np.nan] * len(gdf)
        except Exception as e:
            print(f"错误: 处理栅格 {target_raster} (来源: {source_name}) 时发生未知错误: {e}。为 {source_name} 填充 NaN。")
            # 为这个出错的数据源的所有目标列填充 NaN (假设性填充)
            num_bands = 1 # 无法确定波段数，假设1
            if bands_to_extract: process_bands_indices = [b - 1 for b in bands_to_extract if b>0]
            else: process_bands_indices = [0]
            for band_idx in process_bands_indices:
                  band_num = band_idx + 1
                  col_name = f"{source_name}_b{band_num}" if (bands_to_extract and len(bands_to_extract)>1) or num_bands > 1 else source_name
                  output_data[col_name] = [np.nan] * len(gdf)


    # --- 保存当前月份包含所有提取特征的结果为 CSV 文件 ---
    # 构建输出 CSV 文件的完整路径
    output_csv = OUTPUT_DIR / f"Extracted_Features_{YEAR}_{month_str}.csv"
    try:
        # 将存储结果的字典转换为 Pandas DataFrame
        output_df = pd.DataFrame(output_data)
        # 将 DataFrame 保存为 CSV 文件
        # index=False 表示不将 DataFrame 的索引写入 CSV 文件
        # encoding='utf-8-sig' 有助于在 Excel 中正确显示中文和特殊字符
        output_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"已将 {YEAR}-{month_str} 的提取特征保存至 CSV: {output_csv}")
    except Exception as e:
        # 捕获保存文件时可能出现的错误
        print(f"错误: 保存输出 CSV 文件 {output_csv} 时出错: {e}")

# 所有月份处理完毕
print(f"{YEAR}年所有月份处理完成。")