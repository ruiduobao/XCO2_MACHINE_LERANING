# step1_中国范围数据转为栅格数据

import os
from osgeo import gdal, ogr, osr
import math

# --- 输入参数 ---
vector_path = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\矢量数据\中国区域矢量\China_WGS84.gpkg"
output_raster_path = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\矢量数据\中国区域矢量\China_rasterized_0.05deg.tif" # 输出栅格路径
pixel_size = 0.05  # 分辨率（度）
no_data_value = -9999 # NoData 值
burn_value = 1      # 矢量内部栅格的值
output_format = "GTiff" # 输出格式
target_epsg = 4326    # 目标坐标系 WGS84

# --- GDAL 创建选项 ---
# LZW 压缩, Tiled (256x256)
creation_options = [
    'COMPRESS=LZW',
    'TILED=YES',
    'BLOCKXSIZE=256',
    'BLOCKYSIZE=256'
]

# --- 1. 打开矢量数据 ---
vector_ds = ogr.Open(vector_path)
if vector_ds is None:
    print(f"错误：无法打开矢量文件: {vector_path}")
    exit(1)

vector_layer = vector_ds.GetLayer()
# 获取矢量范围 (minX, maxX, minY, maxY)
x_min, x_max, y_min, y_max = vector_layer.GetExtent()
print(f"矢量范围: X({x_min}, {x_max}), Y({y_min}, {y_max})")

# --- 2. 计算输出栅格尺寸 ---
# 根据范围和分辨率计算栅格的行数和列数
cols = math.ceil((x_max - x_min) / pixel_size)
rows = math.ceil((y_max - y_min) / pixel_size)
print(f"计算栅格尺寸: {cols} 列 x {rows} 行")

# --- 3. 设置地理转换参数 ---
# (左上角X, X方向像素宽度, X方向旋转, 左上角Y, Y方向旋转, Y方向像素高度)
# 注意：Y方向像素高度通常为负值
geotransform = (x_min, pixel_size, 0, y_max, 0, -pixel_size)

# --- 4. 设置空间参考系统 (WGS84) ---
srs = osr.SpatialReference()
srs.ImportFromEPSG(target_epsg)
target_wkt = srs.ExportToWkt()

# --- 5. 创建输出栅格文件 ---
driver = gdal.GetDriverByName(output_format)
target_ds = driver.Create(
    output_raster_path,
    cols,
    rows,
    1,  # 波段数
    gdal.GDT_Int32, # 数据类型 (选择可以容纳 -9999 和 1 的类型)
    options=creation_options
)

if target_ds is None:
    print(f"错误：无法创建输出栅格文件: {output_raster_path}")
    vector_ds = None # 清理
    exit(1)

target_ds.SetGeoTransform(geotransform)
target_ds.SetProjection(target_wkt)

# --- 6. 初始化栅格并进行栅格化 ---
band = target_ds.GetRasterBand(1)
band.SetNoDataValue(no_data_value)
band.Fill(no_data_value) # 首先用 NoData 值填充整个栅格

# 执行栅格化 (将矢量 '烧录' 到栅格)
# gdal.RasterizeLayer(输出数据集, [输出波段列表], 输入图层, burn_values=[烧录值列表])
err = gdal.RasterizeLayer(target_ds, [1], vector_layer, burn_values=[burn_value])

if err != 0:
    print(f"错误：栅格化过程中发生错误，错误代码: {err}")
else:
    print("栅格化成功！")

# --- 7. 清理和关闭 ---
band.FlushCache() # 确保所有写入完成
target_ds = None  # 关闭文件
vector_ds = None  # 关闭输入文件

print(f"栅格文件已保存至: {output_raster_path}")