# step4.2_获取每一行的各要素的数据
import pandas as pd
import rasterio
import os

# 1. 加载 CSV 表格
csv_path = 'E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\卫星数据\网格XCO2加权统计_按年月.csv'
df = pd.read_csv(csv_path)

# 2. 定义栅格文件根目录
root_dir = 'E:\地理所\论文\中国XCO2论文_2025.04\数据'

# 定义类型到文件夹名称的映射
type_to_folder = {
    'Lantitude': '纬度栅格',
    'Longtitude': '经度栅格',
    'UnixTime': '每月时间戳的栅格数据',
    'aspect': '坡向数据',
    'slope':'坡度数据',
    'DEM':'DEM',
    'VIIRS':'夜光遥感',
    'ERA5Land':'ERA5',
    'AOD':'气溶胶厚度',
    'CT2019B':'carbon_tracer',
    'landscan':'landscan',
    'odiac1km':'odiac',
    'humanfootprint':'人类足迹数据',
    'OCO2GEOS':'OCO2_GEOS_XCO2同化数据',
    'CAMStcco2':'CAMS',
    'CLCD':'CLCD',
    'MODISLANDCOVER':'modis_landcover',
    'MOD13A2':'NDVI',
}

# 定义按年份和按年月的数据类型
annual_types = ['Lantitude','Longtitude','aspect','slope','DEM','landscan','humanfootprint','CLCD', 'MODISLANDCOVER'] 

monthly_types = ['UnixTime', 'VIIRS', 'ERA5Land', 'AOD', 'CT2019B', 'odiac1km', 'OCO2GEOS', 'CAMStcco2', 'MOD13A2']
                 
                 
types = list(type_to_folder.keys())  # 使用映射中的所有类型

# 3. 遍历表格每一行，提取像素值
for index, row in df.iterrows():
    year = row['year']
    month = row['month']
    X = int(row['X'])  # 确保 X 是整数（行列索引）
    Y = int(row['Y'])  # 确保 Y 是整数（行列索引）
    
    # 遍历每种类型
    for type_name in types:
        # 根据类型选择文件名格式
        if type_name in annual_types:
            # 按年份的数据，文件名不含月份
            tif_filename = f"{type_name}_{year}.tif"
        else:
            # 按年月的数据，文件名含年份和月份
            tif_filename = f"{type_name}_{year}_{month:02d}.tif"  # 月份格式化为两位数
        
        # 构造栅格文件路径，使用映射中的文件夹名称
        folder_name = type_to_folder[type_name]
        tif_path = os.path.join(root_dir, folder_name, tif_filename)
        
        # 检查文件是否存在
        if not os.path.exists(tif_path):
            print(f"警告：文件 {tif_path} 不存在，跳过")
            continue
        
        # 使用 rasterio 打开栅格文件
        with rasterio.open(tif_path) as src:
            # 获取波段数量
            num_bands = src.count
            
            # 遍历每个波段
            for band in range(1, num_bands + 1):
                # 读取波段数据
                band_data = src.read(band)
                
                # 检查 X 和 Y 是否在栅格范围内
                if Y < 0 or Y >= band_data.shape[0] or X < 0 or X >= band_data.shape[1]:
                    print(f"警告：行 {index}, X={X}, Y={Y} 超出 {tif_path} 范围，跳过")
                    pixel_value = None
                else:
                    # 提取像素值
                    pixel_value = band_data[Y, X]
                    
                    # 检查是否为 NoData 值
                    if pixel_value == src.nodata:
                        pixel_value = None
                
                # 将像素值添加到 DataFrame，新列名为 类型_波段
                col_name = f"{type_name}_band{band}"
                df.at[index, col_name] = pixel_value

# 4. 保存处理后的表格
output_path = 'E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\卫星数据\处理后的网格XCO2加权统计.csv'
df.to_csv(output_path, index=False)
print(f"处理完成，结果已保存到 {output_path}")