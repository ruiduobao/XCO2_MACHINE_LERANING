import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
# import datetime # Not needed if using pd.to_datetime

# --- 1. Configuration ---
# (Same as before)
GOSAT_DIR = Path(r"E:\地理所\论文\中国XCO2论文_2025.04\数据\gosat\2018")
OUTPUT_DIR = Path(r"E:\地理所\论文\中国XCO2论文_2025.04\数据\gosat\2018_extracted_csv_filtered_time_debug") # New output dir
OUTPUT_FILENAME = "GOSAT_L2_SWIR_2018_filtered_extracted_time_debug.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE_PATH = OUTPUT_DIR / OUTPUT_FILENAME

LON_PATH = '/Data/geolocation/longitude'
LAT_PATH = '/Data/geolocation/latitude'
XCO2_PATH = '/Data/mixingRatio/XCO2'
TIME_PATH = '/scanAttribute/time'
QUALITY_FLAG_PATH = '/Data/retrievalQuality/totalPostScreeningResult'
CHI2_PATH = '/Data/retrievalQuality/chi2'
DFS_PATH = '/Data/retrievalQuality/CO2DFS'

GOOD_QUALITY_CODE = 0
MAX_CHI2_THRESHOLD = 2.5 # !! Placeholder !!
MIN_DFS_THRESHOLD = 1.0  # !! Placeholder !!

# --- 3. Data Extraction Function ---
def extract_gosat_data(h5_file_path):
    """ Extracts and filters data, includes time debugging. """
    data_list = []
    filename = h5_file_path.name
    # print(f"Processing file: {filename}")
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # --- Read required data ---
            try:
                lon = f[LON_PATH][:]
                lat = f[LAT_PATH][:]
                xco2 = f[XCO2_PATH][:]
                time_bytes_array = f[TIME_PATH][:]
                quality_flag = f[QUALITY_FLAG_PATH][:]
                chi2_val = f[CHI2_PATH][:]
                co2_dfs_val = f[DFS_PATH][:]
            except KeyError as e:
                print(f"错误: 在文件 {filename} 中找不到必要的数据路径: {e}。跳过此文件。")
                return []

            # --- Basic shape check ---
            if not (lon.shape == lat.shape == xco2.shape == quality_flag.shape == chi2_val.shape == co2_dfs_val.shape and lon.shape[0] == time_bytes_array.shape[0]):
                 print(f"错误: 文件 {filename} 中数据维度不一致。跳过此文件。")
                 return []
            if lon.ndim == 0 or lon.shape[0] == 0: return []
            num_records = lon.shape[0]

            # --- Define and filter invalid values ---
            invalid_val_float = -9999.0
            valid_mask = np.ones(num_records, dtype=bool)
            valid_mask &= ~np.isclose(lon, invalid_val_float)
            valid_mask &= ~np.isclose(lat, invalid_val_float)
            valid_mask &= ~np.isclose(xco2, invalid_val_float)
            valid_mask &= ~np.isclose(chi2_val, invalid_val_float)
            valid_mask &= ~np.isclose(co2_dfs_val, invalid_val_float)
            # ... quality flag invalid check ...
            num_pass_invalid = np.sum(valid_mask)
            if num_pass_invalid == 0: return []

            # --- Apply invalid value mask ---
            lon_valid = lon[valid_mask]
            lat_valid = lat[valid_mask]
            xco2_valid = xco2[valid_mask]
            time_bytes_valid = time_bytes_array[valid_mask]
            quality_flag_valid = quality_flag[valid_mask]
            chi2_valid = chi2_val[valid_mask]
            co2_dfs_valid = co2_dfs_val[valid_mask]

            # --- !! IMPLEMENT QUALITY FILTERING BASED ON DOCUMENTATION !! ---
            quality_good_mask = np.ones(lon_valid.shape, dtype=bool)
            quality_good_mask &= (quality_flag_valid == GOOD_QUALITY_CODE)
            quality_good_mask &= (chi2_valid < MAX_CHI2_THRESHOLD)
            quality_good_mask &= (co2_dfs_valid > MIN_DFS_THRESHOLD)

            # --- Apply the final quality mask ---
            lon_final = lon_valid[quality_good_mask]
            lat_final = lat_valid[quality_good_mask]
            xco2_final = xco2_valid[quality_good_mask]
            time_bytes_final = time_bytes_valid[quality_good_mask]
            quality_flag_final = quality_flag_valid[quality_good_mask]
            chi2_final = chi2_valid[quality_good_mask]
            co2_dfs_final = co2_dfs_valid[quality_good_mask]

            num_pass_quality = len(lon_final)
            if num_pass_quality == 0: return []

            # --- **** DEBUG: Print first few raw time strings **** --- ### ENABLED ###
            if num_pass_quality > 0:
                # print(f"--- Debug Time Info for {filename} ---") # Reduce noise unless needed
                num_to_print = min(1, num_pass_quality) # Print only the first one per file
                for i in range(num_to_print):
                    try:
                        raw_time_str = time_bytes_final[i].tobytes().decode('utf-8', errors='ignore').strip()
                        print(f"第一个原始时间字符串 ({filename}): '{raw_time_str}'") # Print in Chinese for user
                    except Exception as e_dbg:
                        print(f"解码/打印原始时间字符串时出错 [{i}]: {e_dbg}")
                # print(f"--- End Debug Time Info ---")
            # --- **** END DEBUG **** ---


            # --- Parse Time using pd.to_datetime ---
            # --- !! This section likely needs adjustment based on debug output !! ---
            time_final = []
            parse_errors = 0
            for t_bytes in time_bytes_final:
                try:
                    time_str = t_bytes.tobytes().decode('utf-8', errors='ignore').strip()
                    # Attempt parsing using pandas (more flexible)
                    # If you know the *exact* format from debug output, specify it:
                    # parsed_time = pd.to_datetime(time_str, format='<your_exact_format>', errors='coerce')
                    parsed_time = pd.to_datetime(time_str, errors='coerce') # Default flexible parsing
                    time_final.append(parsed_time)
                    if pd.isna(parsed_time): # Count explicit failures
                        parse_errors += 1
                except Exception: # Catch any other errors during processing
                    time_final.append(pd.NaT)
                    parse_errors += 1

            # if parse_errors > 0: # Report errors per file if they occurred
            #     print(f"  警告: 文件 {filename} 中有 {parse_errors}/{len(time_bytes_final)} 个时间解析失败 (变为 NaT)。")


            # --- Store final filtered data ---
            for i in range(len(lon_final)):
                data_list.append({
                    'longitude': lon_final[i],
                    'latitude': lat_final[i],
                    'xco2': xco2_final[i],
                    'time': time_final[i], # Will be datetime object or NaT
                    'quality_flag': quality_flag_final[i],
                    'chi2': chi2_final[i],
                    'co2_dfs': co2_dfs_final[i]
                })

    except FileNotFoundError:
        print(f"错误: 文件未找到 {h5_file_path}")
    except Exception as e:
        print(f"错误: 处理文件 {filename} 时发生意外错误: {e}")

    return data_list

# --- 4. Main Execution ---
if __name__ == "__main__":
    print(f"开始扫描 GOSAT L2 HDF5 文件于: {GOSAT_DIR}")
    h5_files = list(GOSAT_DIR.glob("*.h5"))

    if not h5_files:
        print(f"错误: 在目录 {GOSAT_DIR} 未找到任何 .h5 文件。")
    else:
        print(f"找到 {len(h5_files)} 个 HDF5 文件待处理。")
        all_extracted_data = []

        for h5_file in tqdm(h5_files, desc="处理 GOSAT 文件"):
            file_data = extract_gosat_data(h5_file)
            if file_data:
                all_extracted_data.extend(file_data)

        if not all_extracted_data:
            print("错误: 未能从任何 HDF5 文件中提取有效且通过质量过滤的数据。")
        else:
            print(f"所有文件处理完毕，列表包含 {len(all_extracted_data)} 个数据点字典。") # This count is before NaT drop

            # --- 转换为 Pandas DataFrame ---
            print("正在将提取的数据转换为 Pandas DataFrame...")
            df = pd.DataFrame(all_extracted_data)

            print("--- DataFrame Info BEFORE time conversion/dropna ---") # Check initial types
            df.info()

            # Convert time column explicitly AFTER creating DataFrame
            # This makes it easier to see NaTs from parsing vs other issues
            print("Converting 'time' column to datetime, coercing errors to NaT...")
            df['time'] = pd.to_datetime(df['time'], errors='coerce')

            print("--- DataFrame Info AFTER time conversion ---")
            df.info() # Check Dtype and non-null count again
            print("--- Time Column NaT Count AFTER conversion ---")
            nat_count = df['time'].isnull().sum()
            print(f"时间列中 NaT (无法解析) 的数量: {nat_count}")

            # Drop rows where time parsing failed (resulting in NaT)
            initial_rows = len(df)
            df.dropna(subset=['time'], inplace=True)
            if len(df) < initial_rows:
                print(f"已移除 {initial_rows - len(df)} 行因时间解析失败的数据 (NaT)。")

            print(f"最终 DataFrame 形状 (after dropna): {df.shape}")
            if df.empty:
                 print("错误: 最终 DataFrame 在移除无效时间后为空。请检查上面打印的原始时间字符串，并修正时间解析逻辑。")
            else:
                # --- 保存为 CSV 文件 ---
                print(f"正在保存 DataFrame 到 CSV 文件: {OUTPUT_FILE_PATH}")
                try:
                    df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
                    print(f"数据成功保存至: {OUTPUT_FILE_PATH}")
                except Exception as e:
                    print(f"错误: 保存 CSV 文件时出错: {e}")