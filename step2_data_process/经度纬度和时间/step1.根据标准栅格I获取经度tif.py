# step1.根据标准栅格I获取经度tif
from osgeo import gdal
import numpy as np
from tqdm import tqdm

# 输入和输出文件路径
input_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\范围数据\标准栅格.tif'  # 请替换为您的输入文件路径
output_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\经度纬度栅格\China_Longtitude.tif'

# 打开输入文件
ds = gdal.Open(input_path)
band = ds.GetRasterBand(1)
nodata = band.GetNoDataValue()  # -9999.0
transform = ds.GetGeoTransform()
c = transform[0]  # 左上角经度: 110.3556970343445
a = transform[1]  # 经度方向像素大小: 8.983152841195215e-05

# 创建输出文件
driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create(
    output_path,
    ds.RasterXSize,  # 70011
    ds.RasterYSize,  # 55462
    1,               # 波段数
    gdal.GDT_Float32,  # 数据类型
    options=['COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'BIGTIFF=YES', 'PREDICTOR=2']
)
out_ds.SetGeoTransform(transform)  # 设置仿射变换
out_ds.SetProjection(ds.GetProjection())  # 设置坐标系
out_band = out_ds.GetRasterBand(1)
out_band.SetNoDataValue(nodata)  # 设置NoData值

# 分块处理
tile_size = 25600
for i in tqdm(range(0, ds.RasterYSize, tile_size)):
    for j in range(0, ds.RasterXSize, tile_size):
        # 计算当前块的实际大小（边缘可能不足256）
        win_xsize = min(tile_size, ds.RasterXSize - j)
        win_ysize = min(tile_size, ds.RasterYSize - i)

        # 读取当前块数据
        data = band.ReadAsArray(j, i, win_xsize, win_ysize)

        # 初始化输出数组，填充NoData值
        out_data = np.full((win_ysize, win_xsize), nodata, dtype=np.float32)

        # 找到非空值区域
        mask = data != nodata
        if np.any(mask):
            # 计算全局列号（j为块的起始列）
            global_cols = np.arange(j, j + win_xsize)
            # 计算经度，取像素中心点
            lons = c + a * (global_cols + 0.5)
            # 保留3位小数
            lons = np.round(lons, 3).astype(np.float32)
            # 获取非空值像素的行列索引
            rows, cols = np.where(mask)
            # 将经度赋值给非空值位置
            out_data[rows, cols] = lons[cols]

        # 写入当前块到输出文件
        out_band.WriteArray(out_data, j, i)

# 关闭数据集
ds = None
out_ds = None