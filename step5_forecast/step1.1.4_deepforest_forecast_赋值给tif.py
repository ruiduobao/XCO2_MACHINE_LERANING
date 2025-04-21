import rasterio
import pandas as pd
import numpy as np
import os
import time

# --- 文件路径定义 ---
SOURCE_RASTER_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\范围数据\标准栅格_WGS84.tif"
CSV_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif\统计_2008_03_deepforest_predictions.csv"
OUTPUT_RASTER_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif_预测XCO2\预测xco2_deepforest_2008_03.tif"

# --- 输出栅格参数 ---
OUTPUT_DTYPE = 'float32'
OUTPUT_NODATA = -9999.0

def create_tif_from_csv(csv_path, source_raster_path, output_tif_path):
    """将CSV中的预测结果赋值给TIF文件"""
    try:
        start_time = time.time()
        
        # 读取 CSV 数据
        print(f"正在读取 CSV 文件: {csv_path} ...")
        df = pd.read_csv(csv_path)
        print(f"成功读取 {len(df)} 行数据。")

        # 检查必需的列是否存在
        required_cols = ['X', 'Y', 'predicted_xco2']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"CSV 文件缺少必需的列: {', '.join(missing_cols)}")
        print("CSV 列检查通过。")
        
        # 打开源栅格
        print(f"正在打开源栅格文件: {source_raster_path} ...")
        with rasterio.open(source_raster_path) as src:
            profile = src.profile.copy()  # 获取源栅格的元数据
            source_data = src.read(1)     # 读取源栅格第一个波段的数据
            print(f"源栅格维度: {source_data.shape} (行, 列)")
            
            # 准备输出栅格的数据数组
            print(f"正在创建输出栅格的数据数组，使用 NoData 值: {OUTPUT_NODATA} ...")
            output_data = np.full(source_data.shape, OUTPUT_NODATA, dtype=OUTPUT_DTYPE)
            
            # 遍历 CSV 行，更新输出栅格数组
            print("正在根据 CSV 数据更新输出栅格的值...")
            updated_count = 0
            skipped_source_not_1 = 0
            skipped_out_of_bounds = 0
            
            # 使用迭代器遍历 DataFrame 行
            for index, row in df.iterrows():
                x = int(row['X'])  # 列号
                y = int(row['Y'])  # 行号
                xco2_value = row['predicted_xco2']
                
                # 检查坐标是否在栅格范围内
                if 0 <= y < source_data.shape[0] and 0 <= x < source_data.shape[1]:
                    # 检查源栅格在该位置的值是否为 1
                    if source_data[y, x] == 1:
                        # 如果是 1，则将 CSV 中的 xco2 值赋给输出数组的对应位置
                        output_data[y, x] = xco2_value
                        updated_count += 1
                    else:
                        # 如果源栅格值不是 1，则跳过
                        skipped_source_not_1 += 1
                else:
                    # 如果坐标超出范围，则跳过
                    skipped_out_of_bounds += 1
            
            print("栅格值更新完成。")
            print(f"  成功更新 {updated_count} 个像元。")
            if skipped_source_not_1 > 0:
                print(f"  跳过 {skipped_source_not_1} 个像元 (因为源栅格对应位置值不为 1)。")
            if skipped_out_of_bounds > 0:
                print(f"  跳过 {skipped_out_of_bounds} 个像元 (因为 CSV 中的 X, Y 坐标超出栅格范围)。")
            
            # 更新输出栅格的元数据
            print("正在更新输出栅格的元数据...")
            profile.update(dtype=OUTPUT_DTYPE, nodata=OUTPUT_NODATA, count=1)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_tif_path), exist_ok=True)
            
            # 写入新的栅格文件
            print(f"正在写入新的栅格文件: {output_tif_path} ...")
            with rasterio.open(output_tif_path, 'w', **profile) as dst:
                dst.write(output_data, 1) # 将更新后的数据写入第一个波段
            
        process_time = time.time() - start_time
        print(f"处理完成，耗时: {process_time:.2f} 秒")
        print(f"新的栅格文件写入成功: {output_tif_path}")
        return True
    
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- 主脚本 ---
if __name__ == "__main__":
    print("=" * 80)
    print("开始 Deep Forest XCO2 预测结果转 TIF")
    print("=" * 80)
    
    # 检查输入文件是否存在
    if not os.path.exists(CSV_PATH):
        print(f"错误: 预测CSV文件未找到 {CSV_PATH}")
        exit()
    if not os.path.exists(SOURCE_RASTER_PATH):
        print(f"错误: 源栅格文件未找到 {SOURCE_RASTER_PATH}")
        exit()
    
    # 执行转换
    success = create_tif_from_csv(CSV_PATH, SOURCE_RASTER_PATH, OUTPUT_RASTER_PATH)
    
    if success:
        print("\n转换成功完成！")
    else:
        print("\n转换过程中发生错误，请检查日志。")
    
    print("=" * 80) 