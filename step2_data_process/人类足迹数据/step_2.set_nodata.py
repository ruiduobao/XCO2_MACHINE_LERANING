import os
import glob
import rasterio
import numpy as np
import warnings

# --- 配置 ---
input_dir = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\人类足迹数据\原始数据_WGS84_仿射'
output_dir = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\人类足迹数据\原始数据_WGS84_仿射_nodata"
output_nodata_value = -9999.0 # 用于替换 NaN 并设置为 NoData 标签的数值

# --- 脚本 ---

print(f"输入目录: {input_dir}")
print(f"输出目录: {output_dir}")
print(f"目标 NoData 值: {output_nodata_value}")
print(f"脚本将查找 NaN 值，将其替换为 {output_nodata_value}，并设置文件的 NoData 标签。")

# 1. 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录已确保存在。")

# 2. 查找所有 TIF 文件
try:
    # 同时查找 .tif 和 .TIF
    tif_files = glob.glob(os.path.join(input_dir, "*.tif")) 
    if not tif_files:
        print(f"错误：在输入目录 '{input_dir}' 中未找到任何 .tif 或 .TIF 文件。")
        exit()
    print(f"找到 {len(tif_files)} 个 .tif/.TIF 文件。")
except Exception as e:
    print(f"错误：访问输入目录或查找文件时出错: {e}")
    exit()

# 3. 逐个处理文件
processed_count = 0
error_count = 0

for tif_file_path in tif_files:
    filename = os.path.basename(tif_file_path)
    output_path = os.path.join(output_dir, filename)

    print(f"  处理中: {filename} ...")

    try:
        # 打开源文件
        with rasterio.open(tif_file_path) as src:
            # 获取元数据 profile
            profile = src.profile.copy()
            # 读取数据 (所有波段) -> 形状为 (bands, height, width)
            data = src.read()

            # 检查数据类型是否为浮点型，因为 NaN 只存在于浮点型数据中
            if not np.issubdtype(data.dtype, np.floating):
                warnings.warn(f"    警告: 文件 '{filename}' 的数据类型不是浮点型 ({data.dtype})。文件中不可能存在 NaN 值。将直接复制文件并设置 NoData 标签。")
                nan_mask = np.zeros_like(data, dtype=bool) # 创建一个全为 False 的掩码
            else:
                # 查找数据中的 NaN 值
                nan_mask = np.isnan(data)

            num_nans = np.sum(nan_mask)

            if num_nans > 0:
                print(f"    找到 {num_nans} 个 NaN 像素。正在替换为 {output_nodata_value}...")
                # 使用布尔掩码将所有 NaN 值替换为目标 NoData 值
                # 注意：这里直接修改了内存中的 data 数组
                data[nan_mask] = output_nodata_value
            else:
                print(f"    未在此文件中找到 NaN 像素。")

            # --- 更新 Profile ---
            # 1. 设置 NoData 标签
            profile['nodata'] = output_nodata_value
            
            # 2. 确保驱动是 GTiff
            profile['driver'] = 'GTiff'
            
            # 3. （可选但推荐）设置压缩和其他选项
            profile.update({
                'compress': 'lzw',
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256,
                'bigtiff': 'YES' # 如果文件可能大于4GB
            })
            
            # 4. 移除可能与浮点数+LZW冲突的predictor设置
            if 'predictor' in profile and profile['dtype'].startswith('float'):
                 profile.pop('predictor', None)

        # --- 写入新文件 ---
        # 使用更新后的 profile 和修改后的 data 数组写入新文件
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data) # 将修改后的数据写入

        print(f"    成功: 已保存文件 '{output_path}'。")
        processed_count += 1

    except rasterio.RasterioIOError as e:
         print(f"    错误: 无法打开或读取文件 '{filename}'。跳过。错误信息: {e}")
         error_count += 1
    except Exception as e:
        print(f"    错误: 处理文件 '{filename}' 时发生未知错误。跳过。错误信息: {e}")
        # import traceback; traceback.print_exc() # 取消注释以获取详细调试信息
        error_count += 1

# 4. 结束总结
print("\n处理完成。")
print(f"成功处理的文件数: {processed_count}")
print(f"失败/跳过的文件数: {error_count}")
if error_count > 0:
    print("请检查上面列出的错误信息。")