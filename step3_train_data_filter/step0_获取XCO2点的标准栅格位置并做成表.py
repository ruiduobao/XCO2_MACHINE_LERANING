# step0_获取gosat XCO2点的标准栅格位置并做成表
import geopandas as gpd
import rasterio
import pandas as pd
import os
import glob

# 定义输入矢量文件所在文件夹和输出 CSV 目录
input_folder = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\XCO2各个卫星\TANSAT"
output_folder = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\卫星数据"
raster_path = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\范围数据\标准栅格_WGS84.tif"

# 列出文件夹中所有 .gpkg 文件
gpkg_files = glob.glob(os.path.join(input_folder, "*.gpkg"))

# 打开栅格参考文件，便于重复使用
with rasterio.open(raster_path) as raster:
    
    # 循环处理每个 gpkg 文件
    for vector_path in gpkg_files:
        try:
            # 获取输入文件名称
            input_basename = os.path.splitext(os.path.basename(vector_path))[0]
            output_csv_filename = f"{input_basename}_pixel_coordinates.csv"
            output_csv_path = os.path.join(output_folder, output_csv_filename)
            print(f"\n处理矢量文件：{vector_path}")
            print(f"参考栅格文件：{raster_path}")
            print(f"输出 CSV 文件：{output_csv_path}")
            
            # 读取矢量数据
            gdf = gpd.read_file(vector_path)
            print(f"成功读取 {len(gdf)} 个点要素。")
            
            # 检查 CRS 是否匹配；不匹配则转换
            if gdf.crs != raster.crs:
                print(f"警告: 矢量数据 CRS ({gdf.crs}) 与栅格数据 CRS ({raster.crs}) 不匹配。转换中...")
                gdf = gdf.to_crs(raster.crs)
                print("CRS 转换完成。")
            
            # 提取所有点的 (x, y) 坐标并转换为像素行列号
            coords = [(pt.x, pt.y) for pt in gdf.geometry]
            pixel_coords = [raster.index(x, y) for x, y in coords]
            if pixel_coords:  # 若列表非空
                rows, cols = zip(*pixel_coords)
            else:
                rows, cols = [], []
            
            # 添加新属性列：X（列号）和 Y（行号）
            gdf['X'] = cols
            gdf['Y'] = rows
            
            # 删除 geometry 列，并准备 DataFrame 输出
            df_output = gdf.drop(columns='geometry')
            
            # 保存为 CSV 文件
            df_output.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            
            print("处理完成，CSV 文件已成功保存。")
        except FileNotFoundError as e:
            print(f"错误：文件未找到 - {e}")
            print("请检查输入文件路径是否正确。")
        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
