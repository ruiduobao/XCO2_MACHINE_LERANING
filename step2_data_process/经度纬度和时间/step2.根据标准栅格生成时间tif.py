import os
from osgeo import gdal
import numpy as np
from tqdm import tqdm
from datetime import datetime, timezone

# --- Configuration ---
# Path to the reference raster (determines grid, extent, nodata mask)
standard_raster_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\范围数据\标准栅格.tif'
# Directory to save the output monthly Unix time rasters
output_dir = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\每月时间戳的栅格数据'
# Year for which to generate monthly timestamps
target_year = 2018
# GDAL creation options (copied from your example, adjust if needed)
gdal_options = ['COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'BIGTIFF=YES', 'PREDICTOR=2']
# Tiling size for processing (adjust based on memory)
tile_size = 25600
# --- End Configuration ---

# --- Create output directory if it doesn't exist ---
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# --- Calculate Unix timestamps for the start of each month in the target year (UTC) ---
monthly_timestamps = {}
print(f"Calculating Unix timestamps for {target_year}...")
for month in range(1, 13):
    # Create datetime object for the 1st day of the month at 00:00:00 UTC
    dt_utc = datetime(target_year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
    # Get the Unix timestamp (integer seconds since epoch)
    timestamp = int(dt_utc.timestamp())
    monthly_timestamps[month] = timestamp
    print(f"  {target_year}-{month:02d}-01 00:00:00 UTC -> {timestamp}")

# --- Open Reference Raster to get properties ---
print(f"\nOpening reference raster: {standard_raster_path}")
ref_ds = gdal.Open(standard_raster_path)
if ref_ds is None:
    print(f"Error: Could not open reference raster file: {standard_raster_path}")
    exit(1)

ref_band = ref_ds.GetRasterBand(1)
nodata_value = ref_band.GetNoDataValue()
transform = ref_ds.GetGeoTransform()
projection = ref_ds.GetProjection()
x_size = ref_ds.RasterXSize
y_size = ref_ds.RasterYSize

# Ensure nodata_value is correctly handled (it might be float)
if nodata_value is None:
    print("Warning: Reference raster does not have a NoData value set. Assuming a value like -9999 for internal mask.")
    # Choose a suitable NoData value if none is defined in the source
    # We'll use -9999 for the output, but need something for the mask
    internal_mask_nodata = -9999.0 # Use a float if ref data is float
else:
    internal_mask_nodata = nodata_value
print(f"Reference raster properties: Size=({x_size}, {y_size}), NoData={internal_mask_nodata}")

# Define the output NoData value (should match the data type GDT_Int32)
output_nodata = -9999 # Using an integer for the output timestamp raster

# --- Process each month ---
for month, unix_timestamp in monthly_timestamps.items():
    output_filename = f"UnixTime_{target_year}_{month:02d}.tif"
    output_path = os.path.join(output_dir, output_filename)
    print(f"\n--- Processing Month {month:02d} ({target_year}) ---")
    print(f"Output file: {output_path}")
    print(f"Timestamp value to fill: {unix_timestamp}")

    # Create the output file for the current month
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_path,
        x_size,
        y_size,
        1,                       # Number of bands
        gdal.GDT_Int32,          # Data type for Unix timestamp (integer)
        options=gdal_options
    )
    if out_ds is None:
        print(f"Error: Could not create output file: {output_path}")
        continue # Skip to the next month

    out_ds.SetGeoTransform(transform)
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)
    # Set the NoData value for the output integer raster
    out_band.SetNoDataValue(output_nodata)

    # Process in tiles/blocks
    print(f"Processing in tiles ({tile_size}x{tile_size})...")
    for i in tqdm(range(0, y_size, tile_size), desc=f"Month {month:02d} Rows"):
        for j in range(0, x_size, tile_size):
            # Calculate current block size
            win_xsize = min(tile_size, x_size - j)
            win_ysize = min(tile_size, y_size - i)

            # Read the corresponding block from the *reference* raster
            # This determines where the valid data pixels are
            ref_data = ref_band.ReadAsArray(j, i, win_xsize, win_ysize)

            # Create the output block, initialized with the output NoData value
            out_data = np.full((win_ysize, win_xsize), output_nodata, dtype=np.int32)

            # Create a mask where the reference data is NOT NoData
            mask = ref_data != internal_mask_nodata
            # Also handle potential NaN values if the input is float
            if np.issubdtype(ref_data.dtype, np.floating):
                 mask = mask & (~np.isnan(ref_data))

            # If there are any valid pixels in this block
            if np.any(mask):
                # Assign the current month's Unix timestamp to the valid pixels
                out_data[mask] = unix_timestamp

            # Write the processed block to the output file
            out_band.WriteArray(out_data, j, i)

    # Flush cache and close the output file for the current month
    out_band.FlushCache()
    out_ds = None
    print(f"Finished processing month {month:02d}.")

# --- Cleanup: Close the reference dataset ---
ref_ds = None
print("\nAll months processed.")