# step4.1_检查数据是否为WGS84，如果不是，则添加坐标系

import rasterio
from rasterio.crs import CRS
import os
import shutil
import glob # Needed for finding files
import warnings
from math import isnan # Needed for robust NaN checking

# Suppress specific rasterio warnings if needed (optional)
# warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# 1. 定义栅格文件根目录 和 目标 CRS
root_dir = r'E:\地理所\论文\中国XCO2论文_2025.04\数据' # Use raw string for paths
target_crs = CRS.from_epsg(4326)
expected_nodata = -9999.0

# 2. 定义包含TIFF文件的子文件夹名称
#    (取自之前 type_to_folder 字典的值)
subfolders_to_scan = [
    '纬度栅格', '经度栅格', '每月时间戳的栅格数据', '坡向数据',
    '坡度数据', 'DEM', '夜光遥感', 'ERA5', '气溶胶厚度',
    'carbon_tracer', 'landscan', 'odiac', '人类足迹数据',
    'OCO2_GEOS_XCO2同化数据', 'CAMS', 'CLCD', 'modis_landcover', 'NDVI'
]

# --- 步骤 3: 扫描子文件夹查找所有 .tif 文件 ---
all_tif_files = []
print(f"正在根目录 '{root_dir}' 下扫描指定的子文件夹...")

for folder_name in subfolders_to_scan:
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(folder_path):
        print(f"  警告：子文件夹 '{folder_path}' 不存在，跳过。")
        continue

    print(f"  扫描文件夹: '{folder_path}'")
    # 使用 glob 查找当前文件夹下所有的 .tif 文件
    found_files = glob.glob(os.path.join(folder_path, '*.tif'))
    # 排除临时文件（如果它们遵循特定模式）
    found_files = [f for f in found_files if '_with_crs.tif' not in os.path.basename(f)]
    all_tif_files.extend(found_files)
    print(f"    找到 {len(found_files)} 个 .tif 文件。")


if not all_tif_files:
    print("错误：在指定的子文件夹中未找到任何 .tif 文件。请检查 root_dir 和 subfolders_to_scan 是否正确。")
    exit() # Exit if no files found

print(f"\n总共找到 {len(all_tif_files)} 个 .tif 文件需要检查。")
print("-" * 30)


# --- 步骤 4: 遍历找到的TIFF文件进行检查和处理 ---
nodata_mismatch_files = []
updated_crs_files = []
failed_files = []

for tif_path in sorted(all_tif_files): # Sort for consistent order
    print(f"正在处理: {tif_path}")
    error_occurred = False # Flag to track if any error happened for this file

    # --- 步骤 4.1: 打开文件，检查 NoData 和 CRS ---
    needs_crs_update = False
    original_meta = None
    original_data = None
    current_crs = None
    nodata_value = None

    try:
        with rasterio.open(tif_path) as src:
            nodata_value = src.nodata
            current_crs = src.crs

            # 检查 NoData 值
            is_nodata_mismatch = False
            if nodata_value is None:
                 # Decide how to handle None nodata. Here, we report it.
                 is_nodata_mismatch = True
                 print(f"  信息：文件的 NoData 值未设置 (None)。")
            elif isinstance(nodata_value, (int, float)) and isnan(nodata_value):
                 # Decide how to handle NaN nodata. Here, we report it.
                 is_nodata_mismatch = True
                 print(f"  信息：文件的 NoData 值是 NaN。")
            elif nodata_value != expected_nodata:
                 is_nodata_mismatch = True

            if is_nodata_mismatch:
                 print(f"  *** 警告：文件的 NoData 值 ({nodata_value}) 不是 {expected_nodata}。")
                 nodata_mismatch_files.append(tif_path) # Record the file

            # 检查 CRS
            if current_crs is None:
                print(f"  信息：文件没有 CRS，准备添加 EPSG:4326...")
                needs_crs_update = True
                # 读取元数据和数据，为后续写入做准备
                original_meta = src.meta.copy()
                original_data = src.read() # Read all bands
            elif current_crs != target_crs:
                 print(f"  信息：文件已有 CRS: {current_crs} (不是目标 EPSG:4326)。")
                 # 当前代码只处理 CRS 为 None 的情况。如果需要，可以在此添加逻辑。
            else:
                 print(f"  信息：文件已有目标 CRS: {current_crs}")

    except rasterio.RasterioIOError as e:
        print(f"  错误：打开或读取文件时出错: {e}")
        error_occurred = True
    except Exception as e:
        print(f"  错误：检查文件时发生未知错误: {e}")
        error_occurred = True

    if error_occurred:
        failed_files.append(f"{tif_path} (检查阶段)")
        print("-" * 30)
        continue # 跳过这个文件继续处理下一个

    # --- 步骤 4.2: 如果需要更新 CRS，则执行写入、删除、重命名 ---
    if needs_crs_update:
        temp_tif_path = tif_path.replace('.tif', '_with_crs.tif')
        update_successful = False
        try:
            meta = original_meta
            if meta is None:
                 print(f"  错误：无法更新CRS，因为未能读取原始元数据。")
                 raise ValueError("原始元数据未读取") # Raise error to be caught below
            meta['crs'] = target_crs
            if 'driver' not in meta or meta['driver'] is None:
                 meta['driver'] = 'GTiff' # Ensure driver is set

            print(f"  创建临时文件 {temp_tif_path} 并写入数据...")
            with rasterio.open(temp_tif_path, 'w', **meta) as dst:
                if original_data is None or original_data.size == 0:
                     print(f"  错误：无法写入临时文件，因为未能读取原始数据。")
                     raise ValueError("原始数据未读取或为空") # Raise error
                dst.write(original_data)
            print(f"  已创建新文件 {temp_tif_path}，并添加 CRS。")

            print(f"  删除原始文件 {tif_path}...")
            os.remove(tif_path)
            print(f"  已删除原始文件。")

            print(f"  重命名 {temp_tif_path} 为 {tif_path}...")
            shutil.move(temp_tif_path, tif_path)
            print(f"  文件已成功更新 CRS 并重命名。")
            updated_crs_files.append(tif_path)
            update_successful = True

        except PermissionError as e:
            print(f"  错误：在更新 CRS 时遇到权限错误: {e}")
            print("  请检查是否有其他程序正在占用该文件。")
        except Exception as e:
            print(f"  错误：更新 CRS 时发生错误: {e}")
        finally:
             # 清理临时文件，无论成功与否，只要它存在
             if not update_successful and os.path.exists(temp_tif_path):
                 try:
                     os.remove(temp_tif_path)
                     print(f"  已清理未成功更新的临时文件 {temp_tif_path}")
                 except OSError as rm_err:
                     print(f"  警告：无法清理临时文件 {temp_tif_path}: {rm_err}")
             # 如果更新过程中出错，记录失败
             if not update_successful:
                 error_occurred = True
                 failed_files.append(f"{tif_path} (更新阶段)")


    print("-" * 30) # Separator between files

# --- 结束处理 ---
print("\n======= 处理结果总结 =======")

if updated_crs_files:
    print(f"\n成功更新 CRS 的文件 ({len(updated_crs_files)}):")
    # for f_path in updated_crs_files:
    #     print(f"- {f_path}") # Uncomment to list all files
else:
    print("\n没有文件需要更新 CRS。")

if nodata_mismatch_files:
    print(f"\nNoData 值不是 {expected_nodata} 的文件 ({len(nodata_mismatch_files)}):")
    for f_path in nodata_mismatch_files:
        print(f"- {f_path}")
else:
    print(f"\n所有成功检查的文件的 NoData 值均为 {expected_nodata} (或未设置/NaN)。")

if failed_files:
     print(f"\n处理失败的文件 ({len(failed_files)}):")
     for f_path in failed_files:
         print(f"- {f_path}")
else:
    print("\n所有文件均处理成功（检查或更新）。")

print("\n脚本执行完毕。")