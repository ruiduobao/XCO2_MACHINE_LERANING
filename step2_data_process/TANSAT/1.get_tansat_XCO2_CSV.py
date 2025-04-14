# --- Imports ---
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
# import datetime # Not needed if using pd.to_datetime

# --- 1. Configuration ---
YEAR_TO_PROCESS = 2018 # <<< Define the year to process
# Input TanSat NetCDF file directory
TANSAT_DIR = Path(r"E:\地理所\论文\中国XCO2论文_2025.04\数据\Tansat\原始数据")
# Output directory for the CSV file
OUTPUT_DIR = Path(r"E:\地理所\论文\中国XCO2论文_2025.04\数据\Tansat\原始数据_extracted_csv") # Output directory
# Output CSV filename (add year and region info)
OUTPUT_FILENAME = f"TanSat_L2_XCO2_lite_{YEAR_TO_PROCESS}_filtered_region.csv"

# --- Spatial Filter Configuration ---
LAT_MIN = 15.0  # 南纬15° -> Should be 北纬15° based on typical China extent
LAT_MAX = 57.0  # 北纬57°
LON_MIN = 70.0  # 东经70° (West boundary)
LON_MAX = 140.0 # 东经140° (East boundary)
# Note: Corrected interpretation of lat/lon for China region

# --- Create output directory if it doesn't exist ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE_PATH = OUTPUT_DIR / OUTPUT_FILENAME

# --- 2. Define Key Features and HDF5 Paths ---
# (Same as before)
KEY_FEATURES = [
    'sounding_id', 'latitude', 'longitude', 'time', 'xco2',
    'xco2_uncertainty', 'aod', 'cod', 'surface_pressure',
    'solar_zenith_angle', 'sensor_zenith_angle',
    'xco2_apriori', 'surface_pressure_apriori'
]
# --- HDF5 Variable Paths ---
# !! Double-check these paths against your actual files !!
SOUNDING_ID_PATH = '/sounding_id'
LAT_PATH = '/latitude'
LON_PATH = '/longitude'
TIME_PATH = '/time' # Note: Likely numerical, needs conversion clarification
XCO2_PATH = '/xco2'
XCO2_UNC_PATH = '/xco2_uncertainty'
AOD_PATH = '/aod'
COD_PATH = '/cod'
SURF_PRES_PATH = '/surface_pressure'
SZA_PATH = '/solar_zenith_angle'
VZA_PATH = '/sensor_zenith_angle' # Variable name based on file structure
XCO2_APRIORI_PATH = '/xco2_apriori'
SURF_PRES_APRIORI_PATH = '/surface_pressure_apriori'

# --- Quality Filtering Criteria (PLACEHOLDERS - MUST BE VERIFIED/ADJUSTED) ---
# !! CONSULT OFFICIAL TanSat L2 PRODUCT DOCUMENTATION FOR RECOMMENDED CRITERIA !!
# Example: Filter based on uncertainty and cloud optical depth
MAX_XCO2_UNCERTAINTY = 2.0  # Example: Keep uncertainty < 2.0 ppm
MAX_COD_THRESHOLD = 0.1    # Example: Keep cloud optical depth < 0.1 (near clear)
MAX_AOD_THRESHOLD = 0.5    # Example: Keep AOD < 0.5

# --- 3. Data Extraction Function ---
def extract_tansat_data(nc_file_path):
    """
    Extracts key features, applies spatial and quality filters from a TanSat L2 file.
    Returns a list of dictionaries for valid, filtered soundings.
    """
    data_list = []
    filename = nc_file_path.name
    try:
        with xr.open_dataset(nc_file_path) as ds:
            # --- Read required data ---
            try:
                # Read data using xarray selection for clarity
                lon = ds[LON_PATH.split('/')[-1]].values # Assumes path maps to variable name
                lat = ds[LAT_PATH.split('/')[-1]].values
                xco2 = ds[XCO2_PATH.split('/')[-1]].values
                time_val = ds[TIME_PATH.split('/')[-1]].values
                sounding_id = ds[SOUNDING_ID_PATH.split('/')[-1]].values
                # Read variables needed for filtering and output
                xco2_unc = ds[XCO2_UNC_PATH.split('/')[-1]].values
                aod_val = ds[AOD_PATH.split('/')[-1]].values
                cod_val = ds[COD_PATH.split('/')[-1]].values
                # Read other key features
                surf_pres = ds[SURF_PRES_PATH.split('/')[-1]].values
                sza = ds[SZA_PATH.split('/')[-1]].values
                vza = ds[VZA_PATH.split('/')[-1]].values
                xco2_ap = ds[XCO2_APRIORI_PATH.split('/')[-1]].values
                surf_pres_ap = ds[SURF_PRES_APRIORI_PATH.split('/')[-1]].values

            except KeyError as e:
                print(f"错误: 在文件 {filename} 中找不到必要的数据变量: {e}。跳过此文件。")
                return []
            except Exception as read_e: # Catch other read errors
                print(f"错误: 读取文件 {filename} 数据时出错: {read_e}。跳过此文件。")
                return []


            # --- Basic shape/null check ---
            if lon.ndim == 0 or lon.size == 0: return [] # Skip if empty
            num_records = lon.size

            # --- 1. Initial Check & Invalid Value Masking (if needed) ---
            # Xarray often handles _FillValue automatically, loading as NaN.
            # We primarily need to ensure critical values (lat, lon, xco2, time) are not NaN.
            # Create mask for valid essential data
            valid_mask = ~np.isnan(lon) & ~np.isnan(lat) & ~np.isnan(xco2) & ~np.isnan(time_val)
            # Also check required filter variables are valid numbers
            valid_mask &= ~np.isnan(xco2_unc)
            valid_mask &= ~np.isnan(cod_val)
            valid_mask &= ~np.isnan(aod_val) # Check AOD if using it for filtering

            num_pass_nan = np.sum(valid_mask)
            if num_pass_nan == 0: return [] # Skip if no points have valid essential data

            # Apply initial NaN mask
            lon_valid = lon[valid_mask]
            lat_valid = lat[valid_mask]
            xco2_valid = xco2[valid_mask]
            time_valid = time_val[valid_mask]
            sounding_id_valid = sounding_id[valid_mask]
            xco2_unc_valid = xco2_unc[valid_mask]
            aod_valid = aod_val[valid_mask]
            cod_valid = cod_val[valid_mask]
            # Apply to other features as well
            surf_pres_valid = surf_pres[valid_mask]
            sza_valid = sza[valid_mask]
            vza_valid = vza[valid_mask]
            xco2_ap_valid = xco2_ap[valid_mask]
            surf_pres_ap_valid = surf_pres_ap[valid_mask]

            # --- 2. Spatial Filter ---
            spatial_mask = (lat_valid >= LAT_MIN) & (lat_valid <= LAT_MAX) & \
                           (lon_valid >= LON_MIN) & (lon_valid <= LON_MAX)

            num_pass_spatial = np.sum(spatial_mask)
            if num_pass_spatial == 0: return [] # Skip if no points in region

            # Apply spatial mask
            lon_region = lon_valid[spatial_mask]
            lat_region = lat_valid[spatial_mask]
            xco2_region = xco2_valid[spatial_mask]
            time_region = time_valid[spatial_mask]
            sounding_id_region = sounding_id_valid[spatial_mask]
            xco2_unc_region = xco2_unc_valid[spatial_mask]
            aod_region = aod_valid[spatial_mask]
            cod_region = cod_valid[spatial_mask]
            surf_pres_region = surf_pres_valid[spatial_mask]
            sza_region = sza_valid[spatial_mask]
            vza_region = vza_valid[spatial_mask]
            xco2_ap_region = xco2_ap_valid[spatial_mask]
            surf_pres_ap_region = surf_pres_ap_valid[spatial_mask]


            # --- 3. Quality Filter ---
            # !! CONSULT OFFICIAL TanSat L2 PRODUCT DOCUMENTATION FOR RECOMMENDED CRITERIA !!
            quality_mask = np.ones(lon_region.shape, dtype=bool) # Start with True

            # Example filter based on uncertainty and cloud optical depth
            quality_mask &= (xco2_unc_region < MAX_XCO2_UNCERTAINTY)
            quality_mask &= (cod_region < MAX_COD_THRESHOLD)
            # Optional AOD filter
            # quality_mask &= (aod_region < MAX_AOD_THRESHOLD)

            num_pass_quality = np.sum(quality_mask)
            if num_pass_quality == 0: return [] # Skip if no points pass quality filter

            # Apply quality mask
            lon_final = lon_region[quality_mask]
            lat_final = lat_region[quality_mask]
            xco2_final = xco2_region[quality_mask]
            time_final = time_region[quality_mask]
            sounding_id_final = sounding_id_region[quality_mask]
            xco2_unc_final = xco2_unc_region[quality_mask]
            aod_final = aod_region[quality_mask]
            cod_final = cod_region[quality_mask]
            surf_pres_final = surf_pres_region[quality_mask]
            sza_final = sza_region[quality_mask]
            vza_final = vza_region[quality_mask]
            xco2_ap_final = xco2_ap_region[quality_mask]
            surf_pres_ap_final = surf_pres_ap_region[quality_mask]


            # --- Store final filtered data ---
            for i in range(len(lon_final)):
                data_list.append({
                    'sounding_id': sounding_id_final[i],
                    'latitude': lat_final[i],
                    'longitude': lon_final[i],
                    'time': time_final[i], # Keep numerical time for now
                    'xco2': xco2_final[i],
                    'xco2_uncertainty': xco2_unc_final[i],
                    'aod': aod_final[i],
                    'cod': cod_final[i],
                    'surface_pressure': surf_pres_final[i],
                    'solar_zenith_angle': sza_final[i],
                    'sensor_zenith_angle': vza_final[i],
                    'xco2_apriori': xco2_ap_final[i],
                    'surface_pressure_apriori': surf_pres_ap_final[i]
                })

    except FileNotFoundError:
        print(f"错误: 文件未找到 {nc_file_path}")
    except Exception as e:
        print(f"错误: 处理文件 {filename} 时发生意外错误: {e}")

    return data_list

# --- 4. Main Execution ---
if __name__ == "__main__":
    print(f"开始扫描 TanSat L2 NetCDF 文件于: {TANSAT_DIR} (仅限 {YEAR_TO_PROCESS} 年)")
    # Use glob to find only files matching the year pattern
    # Adjust pattern if filenames differ, e.g., TanSat_..._YYYYMMDD_...nc
    file_pattern = f"*_{YEAR_TO_PROCESS}????.nc" # Assumes YYYYMMDD at end before .nc
    print(f"使用文件模式: {file_pattern}")
    nc_files = sorted(list(TANSAT_DIR.glob(file_pattern))) # Sort files chronologically

    if not nc_files:
        print(f"错误: 在目录 {TANSAT_DIR} 未找到任何匹配 {YEAR_TO_PROCESS} 年的 .nc 文件 (模式: {file_pattern})。")
    else:
        print(f"找到 {len(nc_files)} 个 {YEAR_TO_PROCESS} 年的 NetCDF 文件待处理。")
        all_extracted_data = []

        for nc_file in tqdm(nc_files, desc=f"处理 {YEAR_TO_PROCESS} 年 TanSat 文件"):
            file_data = extract_tansat_data(nc_file)
            if file_data:
                all_extracted_data.extend(file_data)

        if not all_extracted_data:
            print(f"错误: 未能从 {YEAR_TO_PROCESS} 年的文件中提取任何通过筛选的数据。")
        else:
            print(f"\n{YEAR_TO_PROCESS} 年文件处理完毕，共提取 {len(all_extracted_data)} 个有效数据点 (位于指定区域并初步通过质量筛选)。")

            # --- 转换为 Pandas DataFrame ---
            print("正在将提取的数据转换为 Pandas DataFrame...")
            combined_df = pd.DataFrame(all_extracted_data)
            print(f"最终 DataFrame 形状: {combined_df.shape}")

            # --- **** Potential Time Conversion Step **** ---
            # Still needed: Check TanSat documentation for 'time' variable units/epoch
            # Example: If time is seconds since 2000-01-01 00:00:00
            # try:
            #     if 'time' in combined_df.columns:
            #          combined_df['time'] = pd.to_datetime(combined_df['time'], unit='s', origin='2000-01-01', errors='coerce')
            #          print("'time' column converted to datetime.")
            #          combined_df.dropna(subset=['time'], inplace=True) # Drop if conversion failed
            # except Exception as e_time:
            #      print(f"Warning: Converting 'time' column failed: {e_time}")
            # --- **** End Potential Time Conversion Step **** ---

            if combined_df.empty:
                 print("错误: 最终 DataFrame 为空，无法保存 CSV 文件。")
            else:
                # --- 保存为 CSV 文件 ---
                print(f"正在保存 DataFrame 到 CSV 文件: {OUTPUT_FILE_PATH}")
                try:
                    combined_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
                    print(f"数据成功保存至: {OUTPUT_FILE_PATH}")
                except Exception as e:
                    print(f"错误: 保存 CSV 文件时出错: {e}")