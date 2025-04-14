# 将OCO2的矢量拆分每个月一次矢量
import geopandas as gpd
import pandas as pd
import os

# 设置 GDAL 和 PROJ 环境变量
os.environ['GDAL_DATA'] = r'F:\anaconda\envs\geopanadas_rasterio_env\Library\share\gdal'
os.environ['PROJ_LIB'] = r'F:\anaconda\envs\geopanadas_rasterio_env\Library\share\proj'
# 设置输入文件路径
file_path = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\Tansat\原始数据_extractedcsv_转为shp\tansat_2018.shp"

# 读取 GPKG 文件
gdf = gpd.read_file(file_path)


# 按月份分组并保存为单独的 GPKG 文件
for month, group in gdf.groupby('month'):
    # 构建输出文件名，例如 "2018_XCO2_01.gpkg" 表示1月份
    output_file = f"E:\地理所\论文\中国XCO2论文_2025.04\数据\Tansat\原始数据_extractedcsv_转为shp_splitmonth\\2018_XCO2_{int(month):02d}.gpkg"
    # 保存为 GPKG 文件
    group.to_file(output_file, driver='GPKG')
    print(f"已保存 {output_file}")