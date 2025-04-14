import rasterio
import numpy as np

def read_upper_left_value(file_path, window_size=1):
    """
    读取 GeoTIFF 文件左上角的像素值。
    
    Parameters:
    -----------
    file_path : str
        GeoTIFF 文件的路径
    window_size : int
        要读取的窗口大小，默认为1（只读取单个像素）
        
    Returns:
    --------
    upper_left_value : float or int
        左上角的像素值
    """
    with rasterio.open(file_path) as src:
        # 定义一个小窗口，只读取左上角
        window = rasterio.windows.Window(col_off=0, row_off=0, 
                                       width=window_size, height=window_size)
        
        # 只读取指定窗口的数据
        data = src.read(1, window=window)
        
        # 检查数据是否为空
        if data.size == 0:
            raise ValueError("GeoTIFF 文件中没有数据。")
        
        # 提取左上角的值
        upper_left_value = data[0, 0]
        
        return upper_left_value

def check_tif_metadata(file_path):
    """
    读取并显示 TIF 文件的元数据信息
    """
    with rasterio.open(file_path) as src:
        # 获取基本元数据
        print("\nTIF 文件元数据:")
        print(f"Driver: {src.driver}")
        print(f"NoData值: {src.nodata}")
        print(f"数据类型: {src.dtypes}")
        print(f"波段数: {src.count}")
        print(f"尺寸: {src.width} x {src.height}")

        # 检查是否为稀疏格式
        print(src.meta)
        is_sparse = 'SPARSE_OK' in src.meta and src.meta['SPARSE_OK'] == 'YES'
        print(f"是否为稀疏格式: {is_sparse}")

        # 获取完整的标签信息
        print("\n标签信息:")
        for tag, value in src.tags().items():
            print(f"{tag}: {value}")
        
        # 获取每个波段的元数据
        print("\n波段元数据:")
        for i in range(1, src.count + 1):
            print(f"\n波段 {i}:")
            for tag, value in src.tags(i).items():
                print(f"{tag}: {value}")

def read_all_bands_upper_left(file_path, window_size=1):
    """
    读取所有波段左上角的像素值。
    """
    with rasterio.open(file_path) as src:
        window = rasterio.windows.Window(col_off=0, row_off=0, 
                                       width=window_size, height=window_size)
        
        # 读取所有波段的数据
        for band in range(1, src.count + 1):
            data = src.read(band, window=window)
            upper_left_value = data[0, 0]
            
            # 获取波段的统计信息
            stats = src.tags(band)
            stats_info = ""
            if stats:
                min_val = stats.get('STATISTICS_MINIMUM', 'N/A')
                max_val = stats.get('STATISTICS_MAXIMUM', 'N/A')
                mean_val = stats.get('STATISTICS_MEAN', 'N/A')
                stats_info = f" (最小值: {min_val}, 最大值: {max_val}, 平均值: {mean_val})"
            
            print(f"波段 {band} 左上角值: {upper_left_value}{stats_info}")

# 示例用法
if __name__ == "__main__":
    dem_path =  r'E:\地理所\论文\中国XCO2论文_2025.04\数据\CLCD\原始数据_众数重采样_setnull\2018_CLCD_0_01deg_mode_NODATA9999.tif'
    
    try:
        # 1. 读取第一个波段的左上角值
        upper_left = read_upper_left_value(dem_path)
        print(f"第一个波段左上角的值为: {upper_left}")
        
        # 2. 检查元数据
        check_tif_metadata(dem_path)
        
        # 3. 读取所有波段的左上角值
        print("\n所有波段的左上角值:")
        read_all_bands_upper_left(dem_path)
        
    except Exception as e:
        print(f"读取 DEM 文件时出错: {e}")