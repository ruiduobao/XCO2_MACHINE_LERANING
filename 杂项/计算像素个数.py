from osgeo import gdal
import numpy as np
from tqdm import tqdm
import sys # 用于在错误时退出
import os

# --- 输入文件路径 ---
input_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\范围数据\标准栅格.tif'
# --- 要排除的值 ---
nodata_value_to_exclude = -9999.0 # GDAL 读取的 NoData 通常是浮点数

# --- 分块大小 (可以根据内存调整) ---
tile_size = 25600 # 使用与您之前代码相似的块大小

# --- 初始化计数器 ---
total_valid_pixels = 0

print(f"开始计算文件中的有效像素: {input_path}")
print(f"排除值为: {nodata_value_to_exclude}")

# --- 打开输入文件 ---
ds = gdal.Open(input_path)
if ds is None:
    print(f"错误：无法打开文件 {input_path}")
    sys.exit(1) # 退出脚本

band = ds.GetRasterBand(1)
x_size = ds.RasterXSize
y_size = ds.RasterYSize

# 检查文件本身是否定义了 NoData 值（可选，但建议）
file_nodata = band.GetNoDataValue()
if file_nodata is not None:
    print(f"文件定义的 NoData 值: {file_nodata}")
    # 如果文件定义的 NoData 与我们要排除的不同，给出提示
    if file_nodata != nodata_value_to_exclude:
        print(f"注意：文件定义的 NoData ({file_nodata}) 与我们要排除的值 ({nodata_value_to_exclude}) 不同。")
        print(f"计算将基于排除值: {nodata_value_to_exclude}")
else:
    print("文件未定义 NoData 值。计算将基于排除值: {nodata_value_to_exclude}")

print(f"栅格尺寸: {x_size} x {y_size}")
print(f"使用瓦片大小: {tile_size} x {tile_size}")

# --- 分块处理 ---
try:
    for i in tqdm(range(0, y_size, tile_size), desc="处理行"):
        for j in range(0, x_size, tile_size):
            # 计算当前块的实际大小（边缘可能不足 tile_size）
            win_xsize = min(tile_size, x_size - j)
            win_ysize = min(tile_size, y_size - i)

            # 读取当前块数据
            data_block = band.ReadAsArray(j, i, win_xsize, win_ysize)

            # 创建掩模，标记不等于排除值的像素
            mask = data_block != nodata_value_to_exclude

            # 如果数据是浮点类型，还需要排除 NaN 值
            if np.issubdtype(data_block.dtype, np.floating):
                mask = mask & (~np.isnan(data_block))

            # 计算当前块中有效像素的数量
            count_in_block = np.sum(mask)

            # 累加到总数
            total_valid_pixels += count_in_block

except Exception as e:
    print(f"\n处理过程中发生错误: {e}")
    total_valid_pixels = "计算出错" # 标记错误

# --- 清理 ---
band = None
ds = None

# --- 输出结果 ---
print("\n计算完成。")
if isinstance(total_valid_pixels, (int, np.integer)):
    print(f"文件 '{os.path.basename(input_path)}' 中不等于 {nodata_value_to_exclude} 的像素总个数为: {total_valid_pixels}")
else:
    print(f"无法完成计算，原因为: {total_valid_pixels}")