import geopandas as gpd
import os
import glob

# （可选）设置 GDAL 和 PROJ 环境变量，根据实际路径配置
os.environ['GDAL_DATA'] = r'F:\anaconda\envs\geopanadas_rasterio_env\Library\share\gdal'
os.environ['PROJ_LIB'] = r'F:\anaconda\envs\geopanadas_rasterio_env\Library\share\proj'

# 输入和输出文件夹路径
input_dir = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\Tansat\原始数据_extractedcsv_转为shp_splitmonth"
output_dir = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\Tansat\原始数据_extractedcsv_转为shp_splitmonth_筛选中国区域"

# 如果输出文件夹不存在，则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取中国区域矢量数据（边界面）
boundary_shp = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\范围数据\中国区域矢量\2000年初省级.shp"
boundary_gdf = gpd.read_file(boundary_shp)
# 合并所有面的几何为统一区域，注意使用 unary_union
boundary_union = boundary_gdf.unary_union

# 查找输入文件夹下所有 gpkg 文件
gpkg_files = glob.glob(os.path.join(input_dir, "*.gpkg"))

# 对每个 gpkg 文件进行处理
for file in gpkg_files:
    print(f"Processing: {file}")
    
    # 读取点数据 gpkg
    gdf_points = gpd.read_file(file)
    
    # 检查投影是否一致，不一致则转换
    if gdf_points.crs != boundary_gdf.crs:
        gdf_points = gdf_points.to_crs(boundary_gdf.crs)
    
    # 筛选出位于边界内部的点（采用 within 筛选）
    filtered = gdf_points[gdf_points.geometry.within(boundary_union)]
    
    # 构建输出文件路径，保持原文件名
    filename = os.path.basename(file)
    output_file = os.path.join(output_dir, filename)
    
    # 如筛选结果非空，则保存，否则打印提示
    if len(filtered) > 0:
        filtered.to_file(output_file, driver='GPKG')
        print(f"Saved {output_file} with {len(filtered)} points.")
    else:
        print(f"No points within boundary found in {file}. Skipping save.")
