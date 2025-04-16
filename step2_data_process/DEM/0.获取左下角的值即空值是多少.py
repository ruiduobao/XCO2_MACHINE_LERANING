import rasterio
import numpy as np

def read_bottom_left_value(file_path):
    """
    读取 GeoTIFF 文件左下角的像素值。
    
    Parameters:
    -----------
    file_path : str
        GeoTIFF 文件的路径
        
    Returns:
    --------
    bottom_left_value : float or int
        左下角的像素值
    """
    with rasterio.open(file_path) as src:
        # 获取图像高度
        height = src.height
        
        # 定义一个窗口，只读取左下角
        window = rasterio.windows.Window(col_off=0, row_off=height-1, 
                                       width=1, height=1)
        
        # 读取指定窗口的数据
        data = src.read(1, window=window)
        
        # 检查数据是否为空
        if data.size == 0:
            raise ValueError("GeoTIFF 文件中没有数据。")
        
        # 获取左下角的值和nodata值
        bottom_left_value = data[0, 0]
        nodata_value = src.nodata
        
        return bottom_left_value, nodata_value

def read_all_bands_bottom_left(file_path):
    """
    读取所有波段左下角的像素值。
    """
    with rasterio.open(file_path) as src:
        height = src.height
        window = rasterio.windows.Window(col_off=0, row_off=height-1, 
                                       width=1, height=1)
        
        # 获取nodata值
        nodata_value = src.nodata
        print(f"\nNoData值: {nodata_value}")
        
        # 读取所有波段的数据
        for band in range(1, src.count + 1):
            data = src.read(band, window=window)
            bottom_left_value = data[0, 0]
            
            # 获取波段的统计信息
            stats = src.tags(band)
            stats_info = ""
            if stats:
                min_val = stats.get('STATISTICS_MINIMUM', 'N/A')
                max_val = stats.get('STATISTICS_MAXIMUM', 'N/A')
                mean_val = stats.get('STATISTICS_MEAN', 'N/A')
                stats_info = f" (最小值: {min_val}, 最大值: {max_val}, 平均值: {mean_val})"
            
            print(f"波段 {band} 左下角值: {bottom_left_value}{stats_info}")

# 示例用法
if __name__ == "__main__":
    dem_path = r"E:\帮别人\刘南健\gpp2food\AVHRR_GPP_TIFF_monthly\GLASS12B02.V40.A1982_02_Monthly_GPP.tif" 
    
    try:
        # 1. 读取第一个波段的左下角值和nodata值
        bottom_left, nodata = read_bottom_left_value(dem_path)
        print(f"第一个波段左下角的值为: {bottom_left}")
        print(f"NoData值为: {nodata}")
        
        # 2. 读取所有波段的左下角值
        print("\n所有波段的左下角值:")
        read_all_bands_bottom_left(dem_path)
        
    except Exception as e:
        print(f"读取文件时出错: {e}")