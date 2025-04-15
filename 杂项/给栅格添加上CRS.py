import rasterio
from rasterio.crs import CRS

# 输入和输出文件路径
input_path = r'E:/地理所/论文/中国XCO2论文_2025.04/数据/范围数据/标准栅格.tif'
output_path = r'E:/地理所/论文/中国XCO2论文_2025.04/数据/范围数据/标准栅格_WGS84.tif'

# 打开原始栅格文件
with rasterio.open(input_path) as src:
    # 读取栅格数据和元数据
    data = src.read()
    meta = src.meta.copy()
    
    # 设置 CRS 为 WGS84 (EPSG:4326)
    meta['crs'] = CRS.from_epsg(4326)
    
    # 保存为新的 TIFF 文件
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data)

print(f"处理完成，投影后的栅格文件已保存到 {output_path}")