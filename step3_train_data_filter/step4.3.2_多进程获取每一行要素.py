import pandas as pd
import rasterio
import os
import numpy as np # For array_split and NaN
import multiprocessing
from tqdm import tqdm
import time # To measure execution time

# --- Constants and Setup ---
csv_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\卫星数据\网格XCO2加权统计_按年月.csv' # Use raw string
root_dir = r'E:\地理所\论文\中国XCO2论文_2025.04\数据' # Use raw string

# Using the same type_to_folder map, annual/monthly lists, and types list
type_to_folder = {
    'Lantitude': '纬度栅格', 'Longtitude': '经度栅格', 'UnixTime': '每月时间戳的栅格数据',
    'aspect': '坡向数据', 'slope':'坡度数据', 'DEM':'DEM', 'VIIRS':'夜光遥感',
    'ERA5Land':'ERA5', 'AOD':'气溶胶厚度', 'CT2019B':'carbon_tracer',
    'landscan':'landscan', 'odiac1km':'odiac', 'humanfootprint':'人类足迹数据',
    'OCO2GEOS':'OCO2_GEOS_XCO2同化数据', 'CAMStcco2':'CAMS', 'CLCD':'CLCD',
    'MODISLANDCOVER':'modis_landcover', 'MOD13A2':'NDVI',
}
annual_types = ['Lantitude','Longtitude','aspect','slope','DEM','landscan','humanfootprint','CLCD', 'MODISLANDCOVER']
monthly_types = ['UnixTime', 'VIIRS', 'ERA5Land', 'AOD', 'CT2019B', 'odiac1km', 'OCO2GEOS', 'CAMStcco2', 'MOD13A2']
types = list(type_to_folder.keys())

output_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\卫星数据\处理后的网格XCO2加权统计_parallel.csv' # Modified output name
num_processes = 20

# --- Worker Function ---
# This function processes a single chunk of the DataFrame
def process_chunk(df_chunk):
    # Cache for opened raster files within this chunk processing to reduce open/close overhead
    raster_cache = {}
    # Create a copy to avoid SettingWithCopyWarning when modifying the chunk
    chunk_copy = df_chunk.copy()

    try: # Ensure cache is cleaned up
        # Iterate through rows of the specific chunk
        for index, row in chunk_copy.iterrows():
            year = row['year']
            month = row['month']
            # Ensure X, Y are correct integer types for indexing
            try:
                X = int(row['X'])
                Y = int(row['Y'])
            except (ValueError, TypeError):
                 print(f"警告：行 {index} 的 X 或 Y 值无效 ({row['X']}, {row['Y']})，跳过此行。")
                 # Assign NaN to all potential new columns for this row to maintain structure
                 for type_name in types:
                    # Guess max potential bands (e.g., 4) or determine dynamically if possible
                    for band in range(1, 5):
                         col_name = f"{type_name}_band{band}"
                         if col_name not in chunk_copy.columns:
                              chunk_copy[col_name] = np.nan # Add column if needed
                         chunk_copy.loc[index, col_name] = np.nan
                 continue # Skip to next row

            # Process each type for the current row
            for type_name in types:
                # Determine filename based on type (annual or monthly)
                if type_name in annual_types:
                    tif_filename = f"{type_name}_{year}.tif"
                else: # monthly_types
                    tif_filename = f"{type_name}_{year}_{int(month):02d}.tif" # Ensure month is int for formatting

                # Construct full path
                folder_name = type_to_folder[type_name]
                tif_path = os.path.join(root_dir, folder_name, tif_filename)

                # Default value if file missing or error occurs
                pixel_values = {} # Store values for bands {col_name: value}

                if not os.path.exists(tif_path):
                    # print(f"警告：文件 {tif_path} (行 {index}) 不存在，使用 NaN") # Reduce verbosity
                    # Set NaN for potential bands of this missing type
                    for band in range(1, 5): # Assuming max bands again
                        col_name = f"{type_name}_band{band}"
                        pixel_values[col_name] = np.nan
                else:
                    try:
                        # Use cache for rasterio objects
                        if tif_path not in raster_cache:
                            raster_cache[tif_path] = rasterio.open(tif_path)
                        src = raster_cache[tif_path]

                        num_bands = src.count
                        nodata_val = src.nodata

                        for band in range(1, num_bands + 1):
                            col_name = f"{type_name}_band{band}"
                            # Read the band data - needed for every row access here
                            # If performance is still critical, consider optimizing reads further
                            band_data = src.read(band)

                            # Check bounds before accessing pixel
                            if 0 <= Y < band_data.shape[0] and 0 <= X < band_data.shape[1]:
                                value = band_data[Y, X]
                                # Check for NoData (handle float comparison carefully if needed)
                                if nodata_val is not None and np.isclose(value.astype(float), float(nodata_val)):
                                    pixel_values[col_name] = np.nan # Use NaN for NoData
                                else:
                                    pixel_values[col_name] = value
                            else:
                                # print(f"警告：行 {index}, X={X}, Y={Y} 超出 {tif_path} 范围，使用 NaN") # Reduce verbosity
                                pixel_values[col_name] = np.nan # Use NaN for out-of-bounds

                    except Exception as e:
                        print(f"错误：处理行 {index}, 文件 {tif_path}: {e}")
                        # Set NaN for all bands of this type if an error occurs during processing
                        for band in range(1, 5): # Assuming max bands
                            col_name = f"{type_name}_band{band}"
                            pixel_values[col_name] = np.nan

                # Assign collected pixel values to the DataFrame chunk copy
                for col_name, value in pixel_values.items():
                    if col_name not in chunk_copy.columns:
                         # Add column initialized with NaN if it doesn't exist yet
                         chunk_copy[col_name] = np.nan
                    # Use .loc for reliable assignment, especially after creating columns
                    chunk_copy.loc[index, col_name] = value

    finally:
        # Close all cached raster files at the end of processing this chunk
        for src in raster_cache.values():
            src.close()

    return chunk_copy # Return the modified copy of the chunk


# --- Main Execution Block ---
if __name__ == '__main__':
    # Essential for multiprocessing, especially on Windows
    multiprocessing.freeze_support()

    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()

    print(f"加载 CSV 文件: {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"加载完成 {len(df)} 行数据。")

    # Ensure 'year', 'month', 'X', 'Y' are suitable types if not already
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    # X, Y conversion handled within worker function for robustness

    print(f"将 DataFrame 分割成 {num_processes} 块...")
    # Split DataFrame into chunks for parallel processing
    # Note: Chunks might not be perfectly equal in size
    df_chunks = np.array_split(df, num_processes)
    print("分割完成。")

    results = []
    print(f"开始使用 {num_processes} 个进程并行处理...")
    start_process_time = time.time()

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use pool.imap_unordered for potentially better performance and progress tracking
        # tqdm shows progress based on completed chunks
        results_iterator = pool.imap_unordered(process_chunk, df_chunks)

        # Process results as they become available, updating the progress bar
        for result_chunk in tqdm(results_iterator, total=len(df_chunks), desc="处理数据块"):
            results.append(result_chunk)

    print(f"\n并行处理耗时: {time.time() - start_process_time:.2f} 秒。")

    print("正在合并处理结果...")
    # Concatenate the processed chunks back into a single DataFrame
    final_df = pd.concat(results, ignore_index=True)
    # Optional: Sort by original index if needed, though ignore_index=True resets it.
    # If you need original order, don't use ignore_index and sort later.
    print("结果合并完成。")

    print(f"正在保存处理后的表格到: {output_path}...")
    # Save the final DataFrame
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig') # Use utf-8-sig for better Excel compatibility if needed
    end_time = time.time()
    print(f"处理完成！结果已保存。")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {end_time - start_time:.2f} 秒。")