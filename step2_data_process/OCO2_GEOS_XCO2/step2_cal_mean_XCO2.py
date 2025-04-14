import os
import glob
import re
import numpy as np
import rasterio
from collections import defaultdict
import warnings

# --- Configuration ---
input_dir = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\OCO2_GEOS_XCO2同化数据\XCO2_2018_转为tif"
output_dir = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\OCO2_GEOS_XCO2同化数据\XCO2_2018_转为tif_月平均值合成"
output_nodata = -9999.0 # Desired NoData value for the output monthly averages
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
# Regex to extract YYYYMMDD from typical filename patterns
# Adjust if your filenames differ significantly
date_pattern = re.compile(r'(\d{8})') # Looks for any 8-digit sequence

print(f"Found {len(tif_files)} potential .tif files. Grouping by month for year {year_to_process}...")

for f in tif_files:
    basename = os.path.basename(f)
    match = date_pattern.search(basename)
    if match:
        date_str = match.group(1) # YYYYMMDD
        year = date_str[:4]
        month = date_str[4:6] # Extract MM
        
        if year == year_to_process:
            files_by_month[month].append(f)
    else:
         warnings.warn(f"Could not extract YYYYMMDD date from filename: {basename}. Skipping this file.")


if not files_by_month:
     print(f"Error: No files found for the year {year_to_process} matching the expected filename pattern in {input_dir}")
     exit()

print("File grouping complete.")
for month, file_list in sorted(files_by_month.items()):
     print(f"  Month {month}: {len(file_list)} files")


# 3. Process each month
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
            # Use float64 for sum to avoid potential overflow/precision issues
            sum_array = np.zeros(src.shape, dtype=np.float64) 
            count_array = np.zeros(src.shape, dtype=np.int32)

            print(f"  Reading {os.path.basename(first_file)} (1/{len(file_list)})...")
            data = src.read(1) # Read the first band
            
            # Create mask for valid data (not NoData and not NaN)
            if input_nodata is not None:
                 mask = ~np.isnan(data) & (data != input_nodata)
            else: # If input has no nodata defined, only check for NaN
                 mask = ~np.isnan(data)
                 
            # Accumulate sum and count where data is valid
            # Replace potential NoData with 0 before summing where mask is True
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
                    mask = ~np.isnan(data) & (data != input_nodata)
                else:
                    mask = ~np.isnan(data)

                # Accumulate sum and count
                # Replace potential NoData with 0 before summing where mask is True
                sum_array[mask] += np.where(mask, data, 0)[mask]
                count_array[mask] += 1
        except Exception as e:
            print(f"  Error reading file {os.path.basename(filepath)}: {e}. Skipping this file.")
            continue # Skip this file, continue with others for the month

    # 4. Calculate the mean
    print("  Calculating monthly mean...")
    # Initialize mean_array with the desired output NoData value
    # Use float32 for output unless higher precision is strictly needed
    mean_array = np.full(sum_array.shape, output_nodata, dtype=np.float32) 
    
    # Create mask where count is greater than 0 (i.e., at least one valid observation)
    valid_pixels_mask = count_array > 0
    
    # Calculate mean only for valid pixels
    mean_array[valid_pixels_mask] = (sum_array[valid_pixels_mask] / count_array[valid_pixels_mask]).astype(np.float32)

    # 5. Update profile for output
    profile.update({
        'dtype': 'float32',          # Set output data type
        'nodata': output_nodata,     # Set the desired NoData value
        'compress': 'lzw',           # Set LZW compression
        'tiled': True,               # Enable tiling
        'blockxsize': 256,           # Set tile width
        'blockysize': 256,           # Set tile height
        'count': 1                   # Ensure output is single band
    })
    # Remove predictor if not needed for float types (sometimes causes issues)
    if 'predictor' in profile:
         del profile['predictor']


    # 6. Write the output GeoTIFF
    output_filename = f"{year_to_process}_{month}_XCO2_mean.tif"
    output_filepath = os.path.join(output_dir, output_filename)
    
    print(f"  Writing output file: {output_filename}...")
    try:
        with rasterio.open(output_filepath, 'w', **profile) as dst:
            dst.write(mean_array, 1) # Write the calculated mean to the first band
        print(f"  Successfully saved: {output_filepath}")
    except Exception as e:
        print(f"  Error writing output file {output_filename}: {e}")

print("\nProcessing finished.")