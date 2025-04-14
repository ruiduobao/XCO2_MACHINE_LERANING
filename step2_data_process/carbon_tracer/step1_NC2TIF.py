import xarray as xr
import rioxarray # 确保导入以启用 .rio 访问器
import os
import numpy as np # 导入以备不时之需
import warnings

# --- 配置 ---
input_dir = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\carbon_tracer\2018年柱浓度"
output_dir = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\carbon_tracer\2018年柱浓度_tif"
variable_name = "xco2"  # 要提取的变量名 (根据元数据是小写)
target_crs = "EPSG:4326" # 假设为 WGS84 坐标系，适用于经纬度数据

# --- 脚本执行 ---

print(f"输入目录: {input_dir}")
print(f"输出目录: {output_dir}")

# 1. 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
print("输出目录已确保存在。")

# 2. 查找所有 .nc 文件
try:
    nc_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.nc')]
    print(f"在输入目录中找到 {len(nc_files)} 个 .nc 文件。")
except FileNotFoundError:
    print(f"错误：输入目录 '{input_dir}' 不存在。请检查路径。")
    exit()
except Exception as e:
    print(f"错误：访问输入目录时出错: {e}")
    exit()

processed_count = 0
error_count = 0

# 3. 循环处理每个 .nc 文件
for filename in nc_files:
    input_path = os.path.join(input_dir, filename)
    # 构建输出文件名，将 .nc 替换为 .tif
    output_filename = os.path.splitext(filename)[0] + ".tif"
    output_path = os.path.join(output_dir, output_filename)

    print(f"  正在处理: {filename} ...")

    try:
        # 使用 xarray 打开 NetCDF 文件
        # decode_coords="all" 有助于确保所有坐标都被正确解码
        # decode_cf=True (默认) 会自动处理 scale_factor, add_offset, _FillValue 等
        with xr.open_dataset(input_path, decode_coords="all") as ds:

            # 检查变量是否存在
            if variable_name not in ds:
                print(f"    错误: 文件 '{filename}' 中未找到变量 '{variable_name}'。跳过。")
                error_count += 1
                continue

            data_var = ds[variable_name]

            # 处理时间维度 (根据元数据，time 维度大小为 1)
            if 'time' in data_var.dims:
                 # 检查时间维度大小是否为 1
                 if ds.dims['time'] == 1:
                      # 选择第一个时间步并移除 time 维度/坐标
                      data_var = data_var.isel(time=0, drop=True)
                 else:
                      # 如果时间维度不为1，发出警告并跳过（或根据需要采取其他措施）
                      warnings.warn(f"文件 '{filename}' 中的时间维度大小不为 1 ({ds.dims['time']})。此脚本仅处理时间维度为 1 的文件。跳过。")
                      error_count += 1
                      continue
            # 如果没有 time 维度，假定它已经是 2D (latitude, longitude)

            # 检查处理后是否为二维数据
            if len(data_var.dims) != 2:
                 warnings.warn(f"变量 '{variable_name}' 在处理时间维度后不是二维数据。维度: {data_var.dims}。跳过。")
                 error_count += 1
                 continue

            # 设置空间维度名称 (rioxarray 通常能自动识别 'latitude', 'longitude')
            # 显式设置更保险
            try:
                 # 注意 NetCDF 中的维度名称是 'latitude', 'longitude'
                 data_var = data_var.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude', inplace=True)
            except Exception as e:
                 warnings.warn(f"设置空间维度时出错: {e}。尝试继续，但可能需要手动调整坐标名称。")
                 # 如果坐标名称不是 'latitude', 'longitude'，可能需要重命名：
                 # data_var = data_var.rename({'your_lat_dim_name': 'latitude', 'your_lon_dim_name': 'longitude'})
                 # data_var = data_var.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude', inplace=True)

            # 添加 CRS (坐标参考系统) 信息
            data_var.rio.write_crs(target_crs, inplace=True)

            # rioxarray 通常会自动处理 _FillValue。
            # 检查 _FillValue 是否存在，如果不存在则警告
            if '_FillValue' not in data_var.encoding and '_FillValue' not in data_var.attrs:
                 warnings.warn(f"变量 '{variable_name}' 在文件 '{filename}' 中未找到 '_FillValue' 属性。输出的 TIF 可能没有正确设置 NoData 值。")

            # 将 DataArray 写入 GeoTIFF 文件
            # driver='GTiff' 是默认值，可以省略
            # 可以添加压缩选项，例如: compress='LZW'
            data_var.rio.to_raster(output_path, compress='LZW') # 添加LZW压缩

            print(f"    成功: 已保存为 {output_path}")
            processed_count += 1

    except FileNotFoundError:
        print(f"    错误: 文件 '{input_path}' 未找到（可能在处理过程中被移动或删除）。")
        error_count += 1
    except Exception as e:
        print(f"    错误: 处理文件 '{filename}' 时发生未知错误: {e}")
        error_count += 1

# --- 完成 ---
print("\n处理完成。")
print(f"成功转换文件数: {processed_count}")
print(f"失败/跳过文件数: {error_count}")
if error_count > 0:
     print("请检查上面列出的错误和警告信息。")