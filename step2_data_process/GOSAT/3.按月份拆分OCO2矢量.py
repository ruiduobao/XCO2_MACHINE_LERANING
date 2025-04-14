import geopandas as gpd
import pandas as pd
import os

# 设置 GDAL 和 PROJ 环境变量
os.environ['GDAL_DATA'] = r'F:\anaconda\envs\geopanadas_rasterio_env\Library\share\gdal'
os.environ['PROJ_LIB'] = r'F:\anaconda\envs\geopanadas_rasterio_env\Library\share\proj'

# 设置输入文件路径
file_path = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\gosat\2018_extracted_csv_filtered_time_debug\矢量数据_中国区域gosat\中国区域gosat.shp"

# 读取矢量数据
gdf = gpd.read_file(file_path)

# 将 'time' 列转换为 datetime 格式
gdf['time'] = pd.to_datetime(gdf['time'], format="%Y-%m-%d %H:%M:%S.%f")

# 提取月份信息
gdf['month'] = gdf['time'].dt.month

# 定义输出目录并确保存在
output_dir = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\gosat\2018_extracted_csv_filtered_time_debug\矢量数据_中国区域gosat_splitmonth"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 按月份分组并保存为单独的 GPKG 文件
for month, group in gdf.groupby('month'):
    # 构建输出文件名，例如 "2018_GOSAT_XCO2_01.gpkg" 表示1月份数据
    output_file = os.path.join(output_dir, f"2018_GOSAT_XCO2_{month:02d}.gpkg")
    group.to_file(output_file, driver='GPKG')
    print(f"已保存 {output_file}")
