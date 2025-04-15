# -*- coding: utf-8 -*-
import os
from osgeo import gdal
import numpy as np
from tqdm import tqdm

# --- 配置 ---
# 标准栅格路径 (决定网格、范围、nodata 掩模)
standard_raster_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\范围数据\标准栅格.tif'
# 输出每月栅格的目录
output_dir = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\每月时间戳的栅格数据\月份循环' # 可以修改为你想要的输出目录
# GDAL 创建选项 (可以根据需要调整)
gdal_options = ['COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'BIGTIFF=YES', 'PREDICTOR=2']
# 处理块大小 (根据内存调整)
tile_size = 25600
# 输出栅格的 NoData 值
output_nodata = -9999
# --- 配置结束 ---

# --- 创建输出目录（如果不存在） ---
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录: {output_dir}")

# --- 打开参考栅格获取属性 ---
print(f"\n正在打开参考栅格: {standard_raster_path}")
ref_ds = gdal.Open(standard_raster_path)
if ref_ds is None:
    print(f"错误: 无法打开参考栅格文件: {standard_raster_path}")
    exit(1)

ref_band = ref_ds.GetRasterBand(1)
nodata_value_ref = ref_band.GetNoDataValue() # 参考栅格的 NoData 值
transform = ref_ds.GetGeoTransform()
projection = ref_ds.GetProjection()
x_size = ref_ds.RasterXSize
y_size = ref_ds.RasterYSize

# 确保正确处理参考栅格的 nodata 值 (它可能是浮点数)
internal_mask_nodata = nodata_value_ref
if nodata_value_ref is None:
    print("警告: 参考栅格未设置 NoData 值。将在内部掩模中使用一个假定值（例如，输出 NoData 值）。")
    # 如果源中没有定义 NoData，选择一个合适的值用于创建掩模
    internal_mask_nodata = output_nodata # 使用输出 NoData 值作为内部掩模的判断依据
    # 注意：如果参考栅格是浮点型且包含NaN，也需要处理
elif np.isnan(nodata_value_ref):
     print("警告: 参考栅格的 NoData 值是 NaN。")
     internal_mask_nodata = np.nan # 明确使用 NaN 进行比较

print(f"参考栅格属性: 大小=({x_size}, {y_size}), NoData={internal_mask_nodata}")
print(f"输出栅格将使用的 NoData 值: {output_nodata}")

# --- 处理每个月 ---
for month in range(1, 13):
    month_value = month # 这个月栅格的像素值
    output_filename = f"{month}月.tif" # 使用中文月份命名
    output_path = os.path.join(output_dir, output_filename)
    print(f"\n--- 正在处理月份 {month:02d} ---")
    print(f"输出文件: {output_path}")
    print(f"填充的像素值: {month_value}")

    # 创建当前月份的输出文件
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_path,
        x_size,
        y_size,
        1,                      # 波段数
        gdal.GDT_Int32,         # 数据类型 (Int32 足够存储 1-12 和 -9999)
        options=gdal_options
    )
    if out_ds is None:
        print(f"错误: 无法创建输出文件: {output_path}")
        continue # 跳过处理下一个月

    out_ds.SetGeoTransform(transform)
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)
    # 为输出的整数栅格设置 NoData 值
    out_band.SetNoDataValue(float(output_nodata)) # gdal 通常期望 float 类型的 NoData

    # 分块处理
    print(f"正在分块处理 ({tile_size}x{tile_size})...")
    for i in tqdm(range(0, y_size, tile_size), desc=f"月份 {month:02d} 行"):
        for j in range(0, x_size, tile_size):
            # 计算当前块的大小
            win_xsize = min(tile_size, x_size - j)
            win_ysize = min(tile_size, y_size - i)

            # 从 *参考* 栅格读取对应的块
            # 这决定了有效数据像素的位置
            ref_data = ref_band.ReadAsArray(j, i, win_xsize, win_ysize)

            # 创建输出块, 用输出 NoData 值初始化
            out_data = np.full((win_ysize, win_xsize), output_nodata, dtype=np.int32)

            # 创建掩模，其中参考数据 *不是* NoData
            # 需要考虑参考栅格 NoData 是 NaN 的情况
            if np.isnan(internal_mask_nodata):
                mask = ~np.isnan(ref_data)
            else:
                mask = ref_data != internal_mask_nodata
                # 如果参考数据是浮点型，可能也需要排除 NaN
                if np.issubdtype(ref_data.dtype, np.floating):
                    mask = mask & (~np.isnan(ref_data))

            # 如果此块中有任何有效像素
            if np.any(mask):
                # 将当前月份的值分配给有效像素
                out_data[mask] = month_value

            # 将处理后的块写入输出文件
            out_band.WriteArray(out_data, j, i)

    # 刷新缓存并关闭当前月份的输出文件
    out_band.FlushCache()
    out_ds = None # 显式关闭数据集
    print(f"完成处理月份 {month:02d}。")

# --- 清理: 关闭参考数据集 ---
ref_ds = None # 显式关闭数据集
print("\n所有月份处理完毕。")