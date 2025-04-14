import xarray as xr
import rioxarray # 导入 rioxarray 以启用 .rio 访问器
import os
import numpy as np # 用于处理可能的 FillValue

# --- 用户配置 ---
# 请确保将此路径替换为您 NC 文件的实际路径
nc_file_path = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\CAMS\2018年XCO2柱浓度\data_allhours_sfc.nc"
# 要转换的变量名
variable_name = "tcco2"
# 输出 GeoTIFF 文件的前缀和目录
output_dir = "E:\地理所\论文\中国XCO2论文_2025.04\数据\CAMS\CAMS_geotiffs_2018" # 将在此脚本所在的目录下创建一个名为 output_geotiffs 的文件夹
output_prefix = r"tcco2_timestep_"
# 定义坐标参考系统 (CRS) - 根据经纬度信息，WGS84 (EPSG:4326) 是最常用的
crs = "EPSG:4326"
# --- 结束配置 ---

# 检查并创建输出目录
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    except OSError as e:
        print(f"无法创建目录 {output_dir}: {e}")
        exit() # 如果无法创建目录，则退出

print(f"开始处理 NetCDF 文件: {nc_file_path}")

try:
    # 使用 xarray 打开 NetCDF 文件
    # 使用 'with' 语句确保文件在使用后正确关闭
    with xr.open_dataset(nc_file_path) as ds:

        # 选择我们感兴趣的变量
        if variable_name not in ds:
             raise KeyError(f"变量 '{variable_name}' 在文件中未找到。可用变量: {list(ds.variables)}")

        data_var = ds[variable_name]
        print(f"已选择变量: {variable_name}")
        print(f"变量维度: {data_var.dims}")
        print(f"变量形状: {data_var.shape}")

        # --- 设置地理空间信息 ---
        # 1. 明确告知 rioxarray 哪些维度是空间维度
        #    通常 'latitude' 和 'longitude' 会被自动识别，但显式设置更安全
        data_var = data_var.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
        print(f"已设置空间维度: x='longitude', y='latitude'")

        # 2. 分配坐标参考系统 (CRS)
        #    NC 文件元数据指明了经纬度，因此 WGS84 (EPSG:4326) 是合适的
        data_var = data_var.rio.write_crs(crs, inplace=True)
        print(f"已分配 CRS: {crs}")

        # 3. 检查并处理 _FillValue / missing_value
        #    rioxarray/rasterio 在写入时通常会处理 NaN，但我们可以明确设置 nodata 值
        fill_value = data_var.attrs.get('_FillValue', np.nan) # 从属性获取 _FillValue，默认为 NaN
        if not np.isnan(fill_value):
             data_var = data_var.where(data_var != fill_value) # 将 _FillValue 替换为 NaN
             data_var.rio.write_nodata(np.nan, encoded=True, inplace=True) # 设置 NaN 为 nodata 值
             print(f"已处理 FillValue ({fill_value}), 将使用 NaN 作为 nodata 值。")
        else:
             data_var.rio.write_nodata(np.nan, encoded=True, inplace=True) # 默认使用 NaN 作为 nodata 值
             print(f"未找到明确的 _FillValue 或已为 NaN，将使用 NaN 作为 nodata 值。")

        # --- 循环处理时间戳并写入 GeoTIFF ---
        time_dim_name = 'valid_time' # 从元数据确认时间维度名称
        if time_dim_name not in data_var.dims:
            raise ValueError(f"未在变量中找到预期的时间维度 '{time_dim_name}'。")

        num_timesteps = data_var.sizes[time_dim_name]
        print(f"在维度 '{time_dim_name}' 中找到 {num_timesteps} 个时间戳。")

        for i in range(num_timesteps):
            print(f"\n正在处理时间戳索引 {i}...")

            # 使用 .isel() 按索引选择当前时间步的数据
            # 这将返回一个二维的 DataArray (latitude, longitude)
            data_slice = data_var.isel({time_dim_name: i})

            # 构建输出文件名
            output_filename = os.path.join(output_dir, f"{output_prefix}{i+1}.tif")

            try:
                # 使用 rioxarray 的 to_raster 方法将数据切片写入 GeoTIFF 文件
                # rioxarray 会自动从 DataArray 的坐标和属性中提取地理参考信息
                # 添加压缩选项以减小文件大小 (可选)
                data_slice.rio.to_raster(
                    output_filename,
                    driver="GTiff", # 明确指定 GeoTIFF 驱动
                    compress='LZW'  # 使用 LZW 无损压缩 (可选)
                )
                print(f"成功保存: {output_filename}")
            except Exception as e:
                print(f"错误: 写入时间戳 {i} 的 GeoTIFF 文件失败: {e}")

except FileNotFoundError:
    print(f"错误: 输入文件未找到: {nc_file_path}")
except KeyError as e:
    print(f"错误: {e}")
except ValueError as e:
    print(f"错误: {e}")
except ImportError:
    print("错误: 请确保已安装 xarray, rioxarray, rasterio 和 netCDF4 库。")
    print("可以使用 pip 安装: pip install xarray rioxarray rasterio netCDF4")
except Exception as e:
    print(f"发生意外错误: {e}")

print("\n处理完成。")