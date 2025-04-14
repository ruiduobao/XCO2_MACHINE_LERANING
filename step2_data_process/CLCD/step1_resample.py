import rasterio
from rasterio.warp import reproject, calculate_default_transform
from rasterio.enums import Resampling
import numpy as np
import os
import math

# --- 输入和输出文件路径 ---
input_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\CLCD\原始数据\2018_CLCD.tif'
output_path = r'F:\\2018_CLCD_0_01deg_mode.tif' # 添加分辨率和方法到文件名

# --- 目标分辨率 (度) ---
target_resolution = 0.01

# --- 重采样方法 ---
resampling_method = Resampling.mode # 众数

print(f"输入文件: {input_path}")
print(f"输出文件: {output_path}")
print(f"目标分辨率: {target_resolution} 度")
print(f"重采样方法: {resampling_method.name}")

# 创建输出目录（如果不存在）
os.makedirs(os.path.dirname(output_path), exist_ok=True)

try:
    # --- 1. 打开源数据集 ---
    with rasterio.open(input_path) as src:
        print(f"源文件 CRS: {src.crs}")
        print(f"源文件 分辨率 (x, y): ({abs(src.transform.a)}, {abs(src.transform.e)})")

        print(f"源文件 NoData 值: {src.nodata}") # 注意 CLCD 可能没有预定义的 NoData 值

        # --- 2. 计算目标仿射变换参数和尺寸 ---
        # 使用 calculate_default_transform 来获取基于目标分辨率的变换和尺寸
        # 注意：这里假设源文件的 CRS 是地理坐标系 (如 EPSG:4326)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,          # 源坐标系
            src.crs,          # 目标坐标系 (保持不变)
            src.width,        # 源宽度
            src.height,       # 源高度
            *src.bounds,      # 源文件范围 (left, bottom, right, top)
            resolution=target_resolution # 设置目标分辨率
        )
        print(f"计算得到的目标尺寸 (宽, 高): ({dst_width}, {dst_height})")
        print(f"计算得到的目标仿射变换: {dst_transform}")

        # --- 3. 更新输出文件的 Profile (元数据) ---
        profile = src.profile.copy() # 复制源文件的 profile
        profile.update({
            'crs': src.crs,             # 保持坐标系不变
            'transform': dst_transform, # 更新仿射变换
            'width': dst_width,         # 更新宽度
            'height': dst_height,       # 更新高度
            'dtype': "int8",         # 保持数据类型不变 (通常 CLCD 是 uint8)
            'nodata': src.nodata,       # 保持 NoData 值不变 (如果源文件有定义)
            # 保持或设置压缩和分块，对输出文件有利
            'compress': src.profile.get('compress', 'lzw'), # 继承或使用 LZW
            'tiled': src.profile.get('tiled', True),        # 继承或使用 Tiled
            'blockxsize': src.profile.get('blockxsize', 256),
            'blockysize': src.profile.get('blockysize', 256)
            # BIGTIFF 会由 Rasterio 根据需要自动处理
        })

        print(f"输出文件 Profile: {profile}")

        # --- 4. 执行重采样/重投影 ---
        print("开始执行重采样...")
        with rasterio.open(output_path, 'w', **profile) as dst:
            # 假设 CLCD 是单波段文件，如果多波段需要循环
            if src.count == 1:
                reproject(
                    source=rasterio.band(src, 1),          # 源波段
                    destination=rasterio.band(dst, 1),     # 目标波段
                    src_transform=src.transform,           # 源仿射变换
                    src_crs=src.crs,                       # 源坐标系
                    src_nodata=src.nodata,                 # 源 NoData 值
                    dst_transform=dst_transform,           # 目标仿射变换
                    dst_crs=src.crs,                       # 目标坐标系
                    dst_nodata=src.nodata,                 # 目标 NoData 值
                    resampling=resampling_method           # 使用众数重采样
                    # 可以添加 num_threads=-1 来尝试并行处理，但这取决于 GDAL 配置
                    # num_threads=-1
                )
            else:
                # 如果源文件有多波段，循环处理
                print(f"源文件有 {src.count} 个波段，将逐个处理...")
                for i in range(1, src.count + 1):
                    print(f"  处理波段 {i}...")
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        src_nodata=src.nodata, # 假设所有波段 NoData 相同
                        dst_transform=dst_transform,
                        dst_crs=src.crs,
                        dst_nodata=src.nodata, # 假设所有波段 NoData 相同
                        resampling=resampling_method
                    )

    print("\n重采样处理完成！")
    print(f"输出文件已保存至: {output_path}")

except FileNotFoundError:
    print(f"错误：输入文件未找到 {input_path}")
except Exception as e:
    print(f"处理过程中发生错误: {e}")