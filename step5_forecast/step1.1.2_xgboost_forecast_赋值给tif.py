import rasterio
import pandas as pd
import numpy as np
import os

# --- 文件路径定义 ---
source_raster_path = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\范围数据\标准栅格_WGS84.tif"
csv_path = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif_预测XCO2\统计_2008_03_predictions.csv"
# 注意：您提供的 CSV 文件名是 2008_03，请确认年份和月份是否正确。
output_raster_path = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif_预测XCO2\预测xco2.tif"

# --- 输出栅格参数 ---
# predicted_xco2 通常是浮点数，因此输出栅格需要是浮点类型
output_dtype = 'float32'
# 为输出栅格定义一个 NoData 值 (选择一个不太可能出现在真实数据中的值)
output_nodata = -9999.0

try:
    # --- 1. 读取 CSV 数据 ---
    print(f"正在读取 CSV 文件: {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"成功读取 {len(df)} 行数据。")

    # 检查必需的列是否存在
    required_cols = ['X', 'Y', 'predicted_xco2']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"CSV 文件缺少必需的列: {', '.join(missing_cols)}")
    print("CSV 列检查通过。")

    # 为了快速查找，将 X 和 Y 设置为索引 (可选但推荐，特别是对于大型数据集)
    # df.set_index(['X', 'Y'], inplace=True)
    # print("已将 X, Y 设置为 DataFrame 索引以提高查找效率。")
    # 如果不设置索引，后面查找会稍慢，但对中小型数据影响不大

    # --- 2. 打开源栅格 A ---
    print(f"正在打开源栅格文件 (A): {source_raster_path} ...")
    with rasterio.open(source_raster_path) as src:
        profile = src.profile.copy()  # 获取源栅格的元数据（坐标系、变换、尺寸等）
        source_data = src.read(1)     # 读取源栅格第一个波段的数据
        print(f"源栅格 (A) 维度: {source_data.shape} (行, 列)")
        print("源栅格元数据和数据读取成功。")

        # --- 3. 准备输出栅格 B 的数据数组 ---
        print(f"正在创建输出栅格 (B) 的数据数组，使用 NoData 值: {output_nodata} ...")
        # 创建一个与源栅格形状相同、数据类型为 float32 的数组，并用 NoData 值填充
        output_data = np.full(source_data.shape, output_nodata, dtype=output_dtype)

        # --- 4. 遍历 CSV 行，更新输出栅格数组 ---
        print("正在根据 CSV 数据更新输出栅格 (B) 的值...")
        updated_count = 0
        skipped_source_not_1 = 0
        skipped_out_of_bounds = 0

        # 使用迭代器遍历 DataFrame 行，这样更节省内存
        for index, row in df.iterrows():
            x = int(row['X'])  # 列号
            y = int(row['Y'])  # 行号
            xco2_value = row['predicted_xco2']

            # 检查坐标是否在栅格范围内
            if 0 <= y < source_data.shape[0] and 0 <= x < source_data.shape[1]:
                # 检查源栅格 A 在该位置的值是否为 1
                if source_data[y, x] == 1:
                    # 如果是 1，则将 CSV 中的 xco2 值赋给输出数组 B 的对应位置
                    output_data[y, x] = xco2_value
                    updated_count += 1
                else:
                    # 如果源栅格值不是 1，则跳过（输出栅格保持 NoData 值）
                    skipped_source_not_1 += 1
            else:
                # 如果坐标超出范围，则跳过
                skipped_out_of_bounds += 1

            # 可以添加一个进度指示器，如果 CSV 文件很大
            # if (index + 1) % 5000 == 0:
            #    print(f"  已处理 {index + 1}/{len(df)} 行...")

        print("栅格值更新完成。")
        print(f"  成功更新 {updated_count} 个像元。")
        if skipped_source_not_1 > 0:
            print(f"  跳过 {skipped_source_not_1} 个像元 (因为源栅格对应位置值不为 1)。")
        if skipped_out_of_bounds > 0:
            print(f"  跳过 {skipped_out_of_bounds} 个像元 (因为 CSV 中的 X, Y 坐标超出栅格范围)。")


        # --- 5. 更新输出栅格 B 的元数据 ---
        print("正在更新输出栅格 (B) 的元数据...")
        profile.update(dtype=output_dtype, nodata=output_nodata, count=1)
        print(f"  数据类型设置为: {output_dtype}")
        print(f"  NoData 值设置为: {output_nodata}")

        # --- 6. 写入新的栅格文件 B ---
        print(f"正在写入新的栅格文件 (B): {output_raster_path} ...")
        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(output_data, 1) # 将更新后的数据写入第一个波段
        print("新的栅格文件 (B) 写入成功！")

except FileNotFoundError as e:
    print(f"错误：文件未找到 - 请检查路径 '{e.filename}' 是否正确。")
except ValueError as e:
    print(f"错误：数据或配置问题 - {e}")
except KeyError as e:
    print(f"错误: 在 CSV 文件中找不到列 '{e}'。请确保 CSV 包含 'X', 'Y', 和 'predicted_xco2' 列。")
except Exception as e:
    print(f"处理过程中发生未知错误: {e}")
    import traceback
    traceback.print_exc()