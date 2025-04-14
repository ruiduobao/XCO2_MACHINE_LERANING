import os
import glob
import re
import numpy as np
import rasterio
from collections import defaultdict
import warnings

# --- Configuration ---
# 输入目录：包含转换后的日平均 TIF 文件
input_dir = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\carbon_tracer\2018年柱浓度_tif"
# 输出目录：存放计算出的月平均 TIF 文件
output_dir = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\carbon_tracer\2018年柱浓度_tif_月平均值合成"
# 输出月平均文件的 NoData 值
output_nodata = -9999.0
# 要处理的年份
year_to_process = "2018"

# --- Script ---

print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")

# 1. Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory ensured.")

# 2. Find all relevant TIF files and group by month
tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
files_by_month = defaultdict(list)

# --- 修改开始 ---
# 修改正则表达式以匹配 YYYY-MM-DD 格式
# (\d{4}) 捕获年份 (YYYY)
# (\d{2}) 捕获月份 (MM)
# (\d{2}) 捕获日期 (DD)
# 使用 \.tif$ 确保匹配文件名末尾的日期（防止匹配到文件名中间的日期）
date_pattern = re.compile(r'(\d{4})-(\d{2})-(\d{2})\.tif$', re.IGNORECASE) # 添加 re.IGNORECASE 使 .tif 不区分大小写
# --- 修改结束 ---

print(f"Found {len(tif_files)} potential .tif files. Grouping by month for year {year_to_process}...")

for f in tif_files:
    basename = os.path.basename(f)
    # 使用修改后的正则表达式搜索
    match = date_pattern.search(basename)
    if match:
        # --- 修改开始 ---
        # 提取年份 (group 1) 和月份 (group 2)
        year = match.group(1) # YYYY
        month = match.group(2) # MM
        # day = match.group(3) # DD (如果需要的话)
        # --- 修改结束 ---

        if year == year_to_process:
            files_by_month[month].append(f)
        # 如果年份不匹配，则不处理 (这里不需要 else 警告，因为可能有其他年份的文件)
    else:
         # 如果文件名不符合 YYYY-MM-DD.tif 格式，给出警告
         warnings.warn(f"Could not extract YYYY-MM-DD date from filename: {basename}. Skipping this file.")


if not files_by_month:
     print(f"Error: No files found for the year {year_to_process} matching the expected filename pattern 'YYYY-MM-DD.tif' in {input_dir}")
     exit()

print("File grouping complete.")
for month, file_list in sorted(files_by_month.items()):
     print(f"  Month {month}: {len(file_list)} files")


# 3. Process each month (这部分代码与之前相同，无需修改)
for month, file_list in sorted(files_by_month.items()):
    if not file_list:
        print(f"Skipping month {month} - no files found.")
        continue

    print(f"\nProcessing Month: {year_to_process}-{month}...")

    sum_array = None
    count_array = None
    profile = None # To store metadata like CRS, transform, dimensions

    # Read the first file to initialize arrays and get profile
    first_file = file_list[0]
    try:
        with rasterio.open(first_file) as src:
            profile = src.profile.copy() # Get metadata
            input_nodata = src.nodata    # Get NoData value from the *input* file

            # Initialize sum and count arrays
            sum_array = np.zeros(src.shape, dtype=np.float64)
            count_array = np.zeros(src.shape, dtype=np.int32)

            print(f"  Reading {os.path.basename(first_file)} (1/{len(file_list)})...")
            data = src.read(1) # Read the first band

            # Create mask for valid data (not NoData and not NaN)
            if input_nodata is not None:
                 # Also check against the intended output nodata just in case it appears in input
                 mask = ~np.isnan(data) & (data != input_nodata) & (data != output_nodata)
            else: # If input has no nodata defined, only check for NaN and output nodata
                 mask = ~np.isnan(data) & (data != output_nodata)

            # Accumulate sum and count where data is valid
            sum_array[mask] += np.where(mask, data, 0)[mask]
            count_array[mask] += 1

    except Exception as e:
        print(f"  Error reading first file {os.path.basename(first_file)}: {e}. Skipping month {month}.")
        continue # Skip to next month if the first file fails

    # Process remaining files for the month
    for i, filepath in enumerate(file_list[1:], start=2):
        print(f"  Reading {os.path.basename(filepath)} ({i}/{len(file_list)})...")
        try:
            with rasterio.open(filepath) as src:
                # Basic check: Ensure dimensions match the first file
                if src.shape != profile['height'] or src.width != profile['width']:
                     warnings.warn(f"    Dimension mismatch for {os.path.basename(filepath)}. Expected ({profile['height']}, {profile['width']}), got ({src.shape[0]}, {src.shape[1]}). Skipping this file.")
                     continue

                input_nodata = src.nodata # Get NoData for *this* file
                data = src.read(1)

                # Create mask for valid data
                if input_nodata is not None:
                     mask = ~np.isnan(data) & (data != input_nodata) & (data != output_nodata)
                else:
                     mask = ~np.isnan(data) & (data != output_nodata)

                # Accumulate sum and count
                sum_array[mask] += np.where(mask, data, 0)[mask]
                count_array[mask] += 1
        except Exception as e:
            print(f"  Error reading file {os.path.basename(filepath)}: {e}. Skipping this file.")
            continue # Skip this file, continue with others for the month

    # 4. Calculate the mean
    print("  Calculating monthly mean...")
    mean_array = np.full(sum_array.shape, output_nodata, dtype=np.float32)
    valid_pixels_mask = count_array > 0
    mean_array[valid_pixels_mask] = (sum_array[valid_pixels_mask] / count_array[valid_pixels_mask]).astype(np.float32)



    # 5. Update profile for output
    profile.update({
        'dtype': 'float32',
        'nodata': output_nodata,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'count': 1
    })
    if 'predictor' in profile:
         del profile['predictor']

    # 6. Write the output GeoTIFF
    # --- 修改输出文件名格式，保持一致性 ---
    output_filename = f"CT2019B_{year_to_process}_{month}_xCO2_mean.tif" # 例如: CT2019B_2018_01_xCO2_mean.tif
    # --- 修改结束 ---
    output_filepath = os.path.join(output_dir, output_filename)

    print(f"  Writing output file: {output_filename}...")
    try:
        with rasterio.open(output_filepath, 'w', **profile) as dst:
            dst.write(mean_array, 1) # Write the calculated mean to the first band
        print(f"  Successfully saved: {output_filepath}")
    except Exception as e:
        print(f"  Error writing output file {output_filename}: {e}")

print("\nProcessing finished.")