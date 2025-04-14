import os
import glob
import rasterio
import warnings

# --- 配置 ---
input_dir = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\气溶胶厚度\1KM"
output_dir = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\气溶胶厚度\1KM_set_WGS84"
target_crs = "EPSG:4326" # WGS84 地理坐标系统对应的 EPSG 代码

# --- 脚本 ---

print(f"输入目录: {input_dir}")
print(f"输出目录: {output_dir}")
print(f"目标指定坐标系: {target_crs} (WGS84)")
print("注意: 此脚本将 *指定* 或 *覆盖* 目标坐标系到输出文件中。")
print("它 *不会* 对栅格数据进行重投影计算。像素值和网格位置保持不变。")
print("如果输入文件已有坐标系信息，该信息将被覆盖！")

# 1. 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录已确保存在。")

# 2. 查找所有相关的 TIF 文件
try:
    # 同时查找 .tif 和 .TIF 后缀
    tif_files = glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.TIF"))
    if not tif_files:
        print(f"错误：在输入目录 '{input_dir}' 中未找到任何 .tif 或 .TIF 文件。请检查路径。")
        exit()
    print(f"找到 {len(tif_files)} 个 .tif/.TIF 文件。")
except Exception as e:
    print(f"错误：访问输入目录或查找文件时出错: {e}")
    exit()


# 3. 逐个处理 TIF 文件
processed_count = 0
error_count = 0

for tif_file_path in tif_files:
    filename = os.path.basename(tif_file_path)
    output_path = os.path.join(output_dir, filename)

    print(f"  处理中: {filename} ...")

    try:
        # 以读取模式打开源文件
        with rasterio.open(tif_file_path) as src:
            # 读取栅格数据（所有波段）
            # 注意：如果文件非常大，这里可能需要分块读写，但对于一般大小的文件，一次性读取是可行的
            data = src.read()
            
            # 获取源文件的元数据 (profile)
            profile = src.profile.copy()
            original_crs = src.crs # 获取原始CRS信息以供检查

            # 检查原始CRS，并给出提示或警告
            if original_crs and original_crs.is_valid:
                # 检查是否已经是目标CRS
                if original_crs == rasterio.crs.CRS.from_string(target_crs):
                    print(f"    信息: 文件 '{filename}' 已具有坐标系 {target_crs}。将直接复制并应用输出设置。")
                else:
                    # 如果存在有效CRS但不是目标CRS，则发出警告
                    warnings.warn(f"    警告: 文件 '{filename}' 的现有坐标系 '{original_crs}' 将被覆盖为 '{target_crs}'。")
            else:
                 # 如果没有有效CRS或CRS无效
                 print(f"    信息: 文件 '{filename}' 没有有效的现有坐标系，将指定为 '{target_crs}'。")

            # 更新元数据：将 CRS 设置为目标 CRS (WGS84)
            profile['crs'] = target_crs
            
            # 确保驱动程序是 GeoTIFF
            profile['driver'] = 'GTiff'
            
            # （可选）添加压缩和其他选项以优化输出
            profile.update({
                'compress': 'lzw',
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256,
                'bigtiff': 'YES' # 如果文件可能超过4GB，则启用BigTIFF
            })
            
            # 对于LZW压缩和浮点数数据，有时需要移除predictor设置
            if 'predictor' in profile and profile['dtype'].startswith('float'):
                profile.pop('predictor', None) # 移除 predictor 键


        # 以写入模式打开目标文件，使用更新后的元数据
        with rasterio.open(output_path, 'w', **profile) as dst:
            # 将原始数据写入新文件
            # 因为 profile['count'] 继承自源文件，所以 data 的波段数会自动匹配
            dst.write(data)

        print(f"    成功: 已保存文件 '{output_path}'，坐标系设置为 {target_crs}.")
        processed_count += 1

    # 捕获特定的 rasterio 错误
    except rasterio.RasterioIOError as e:
         print(f"    错误: 无法打开或读取文件 '{filename}'。跳过。错误信息: {e}")
         error_count += 1
    # 捕获其他所有可能的错误
    except Exception as e:
        print(f"    错误: 处理文件 '{filename}' 时发生未知错误。跳过。错误信息: {e}")
        # 如果需要详细的调试信息，可以取消下面的注释
        # import traceback
        # traceback.print_exc()
        error_count += 1

# 4. 输出最终总结
print("\n处理完成。")
print(f"成功处理的文件数: {processed_count}")
print(f"失败/跳过的文件数: {error_count}")
if error_count > 0:
    print("请检查上面列出的错误信息。")