# step1_nc2tif
import xarray as xr
import rioxarray # 导入 rioxarray 以启用 .rio 访问器
import os
import numpy as np # 虽然此脚本中可能不是必需的，但导入以备不时之需

# --- 配置 ---
# 输入 NetCDF 文件所在的目录
input_dir = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\OCO2_GEOS_XCO2同化数据\XCO2_2018"
# 输出 GeoTIFF 文件要保存的目录
output_dir = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\OCO2_GEOS_XCO2同化数据\XCO2_2018_转为tif"
# 要提取的变量名称
variable_name = "XCO2"
# 指定数据的坐标参考系统 (CRS)，对于经纬度数据，通常是 WGS84 (EPSG:4326)
# 如果你的数据有明确的CRS元数据，rioxarray可能会自动读取，但显式设置更安全
target_crs = "EPSG:4326"

# --- 脚本执行 ---

# 1. 确保输出目录存在，如果不存在则创建
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录 '{output_dir}' 已确保存在。")

# 2. 遍历输入目录中的所有文件
print(f"开始处理目录 '{input_dir}' 中的文件...")
processed_files = 0
error_files = []

for filename in os.listdir(input_dir):
    # 检查文件是否为 NetCDF 文件
    if filename.lower().endswith(".nc4"):
        input_file_path = os.path.join(input_dir, filename)
        # 构建输出文件名 (将 .nc4 替换为 .tif)
        output_filename = os.path.splitext(filename)[0] + ".tif"
        output_file_path = os.path.join(output_dir, output_filename)

        print(f"  正在处理: {filename} ...")

        try:
            # 3. 使用 xarray 打开 NetCDF 文件
            # 使用 with 语句确保文件在使用后正确关闭
            with xr.open_dataset(input_file_path) as ds:

                # 4. 选择所需的变量 (XCO2)
                # 确保变量名与文件中的名称完全匹配
                if variable_name not in ds.variables:
                    print(f"    错误：变量 '{variable_name}' 在文件 {filename} 中未找到。跳过此文件。")
                    error_files.append(filename + " (变量未找到)")
                    continue

                data_var = ds[variable_name]

                # 5. 处理时间维度
                # 根据你提供的元数据，time 维度大小为 1。我们需要选择第一个（也是唯一一个）时间步，
                # 使数据变为二维 (lat, lon) 以便保存为 GeoTIFF。
                if 'time' in data_var.dims:
                    # isel(time=0) 选择第一个时间索引
                    # drop=True 移除选择后的 time 坐标，使其成为纯粹的二维数据
                    data_var = data_var.isel(time=0, drop=True)
                elif len(data_var.dims) != 2:
                     print(f"    警告：变量 '{variable_name}' 在文件 {filename} 中的维度不是预期的 (time, lat, lon) 或 (lat, lon)。尝试继续，但结果可能不正确。维度: {data_var.dims}")


                # 6. 设置空间维度和 CRS (坐标参考系统)
                # rioxarray 通常能自动识别 'lat' 和 'lon'，但显式设置更可靠
                # set_spatial_dims 会告诉 rioxarray 哪些维度是 x 和 y
                try:
                    data_var = data_var.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
                except Exception as e:
                     print(f"    警告：设置空间维度时出错: {e}。尝试继续...")

                # write_crs 将 CRS 信息写入 DataArray
                # 如果数据已经有 CRS 信息，这一步会覆盖它；如果没有，则添加
                data_var.rio.write_crs(target_crs, inplace=True)

                # 7. 设置 NoData 值
                # rioxarray 通常会自动从 _FillValue 或 missing_value 读取 NoData 值
                # 但我们可以显式指定，以防万一
                # 从元数据中获取 _FillValue
                fill_value = ds[variable_name].attrs.get('_FillValue')
                missing_value = ds[variable_name].attrs.get('missing_value')

                nodata_val = fill_value if fill_value is not None else missing_value
                if nodata_val is not None:
                    # 确保nodata_val不是numpy的nan，因为rio.to_raster需要一个具体的数值
                     if not np.isnan(nodata_val):
                           data_var = data_var.rio.write_nodata(nodata_val, encoded=True) # encoded=True 确保写入TIFF元数据
                     else:
                           # 如果原始填充值是 NaN，可以选择一个常用的 NoData 值，如 -9999
                           # 或者让 rioxarray/GDAL 自动处理（可能默认为 NaN 或其他值）
                           print(f"    信息：原始填充值是 NaN，未显式设置 NoData 值。")
                           # data_var = data_var.rio.write_nodata(-9999.0, encoded=True) # 或者选择一个合适的数值


                # 8. 将 DataArray 写入 GeoTIFF 文件
                # compress='LZW' 是一个常用的无损压缩选项
                data_var.rio.to_raster(output_file_path, compress='LZW', driver='GTiff')

                print(f"    成功: 已保存为 {output_file_path}")
                processed_files += 1

        except Exception as e:
            print(f"    错误: 处理文件 {filename} 时发生错误: {e}")
            error_files.append(filename + f" ({e})")

    else:
        # 如果文件不是 .nc4 文件，可以选择忽略或打印消息
        # print(f"  跳过非 .nc4 文件: {filename}")
        pass

# --- 完成 ---
print("\n处理完成。")
print(f"总共处理了 {processed_files} 个 .nc4 文件。")
if error_files:
    print(f"处理过程中遇到错误的文件 ({len(error_files)} 个):")
    for err_file in error_files:
        print(f"  - {err_file}")
else:
    print("所有文件均已成功处理（或跳过）。")