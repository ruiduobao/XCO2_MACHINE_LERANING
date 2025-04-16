import rasterio
import pandas as pd
import numpy as np
import os

# --- 输入和输出配置 ---
raster_path = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\范围数据\标准栅格_WGS84.tif"
# 输出目录保持不变
output_dir = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据"
# 输出文件名的基础部分 (不含年月和扩展名)
output_filename_base = "标准栅格XY"
# 定义目标年份
target_year = 2018

# --- 确保输出目录存在 ---
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建输出目录: {output_dir}")

# --- 读取栅格并获取坐标 (只执行一次) ---
rows_list = []
cols_list = []
points_found = False # 标记是否找到点

try:
    print(f"正在打开栅格文件: {raster_path}")
    with rasterio.open(raster_path) as src:
        print("栅格文件打开成功。")
        # 读取第一个波段
        data = src.read(1)
        print(f"栅格数据维度: {data.shape} (行, 列)")
        print(f"栅格数据类型: {data.dtype}")

        print("正在查找值为 1 的栅格...")
        rows, cols = np.where(data == 1)

        rows_list = rows.tolist()
        cols_list = cols.tolist()
        num_points = len(rows_list)
        print(f"找到 {num_points} 个值为 1 的栅格点。")

        if num_points > 0:
            points_found = True
        else:
             print("警告：在栅格文件中未找到值为 1 的像元。")

        # 可以在这里释放内存，如果栅格数据很大
        del data
        del rows
        del cols

except FileNotFoundError:
    print(f"错误：栅格文件未找到 - {raster_path}")
    print("请检查文件路径是否正确。程序将退出。")
    exit() # 如果找不到栅格，后续无法进行，直接退出
except MemoryError:
    print(f"错误：内存不足。栅格文件可能太大，无法一次性读入内存。")
    print("程序将退出。")
    exit()
except Exception as e:
    print(f"读取栅格过程中发生未知错误: {e}")
    import traceback
    traceback.print_exc()
    print("程序将退出。")
    exit()

# --- 如果找到了点，则为每个月生成 CSV 文件 ---
if points_found:
    print(f"\n开始为年份 {target_year} 生成每月 CSV 文件...")

    # 循环 12 个月
    for month in range(1, 13):
        # 构建当前月份的输出文件名，月份用两位数表示（例如 01, 02, ..., 12）
        output_csv_filename = f"{output_filename_base}_{target_year}_{month:02d}.csv"
        output_csv_path = os.path.join(output_dir, output_csv_filename)

        print(f"  正在生成文件: {output_csv_filename} ...")

        try:
            # 为当前月份创建 DataFrame
            # 直接在这里添加 year 和 month 列
            # Pandas 会自动将单个值扩展到所有行
            df_month = pd.DataFrame({
                'X': cols_list,      # 列号
                'Y': rows_list,      # 行号
                'year': target_year, # 固定年份
                'month': month       # 当前月份
            })

            # (可选) 调整列的顺序，使年月在后面
            df_month = df_month[['X', 'Y', 'year', 'month']]

            # 保存为 CSV 文件
            df_month.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            # print(f"    文件已保存: {output_csv_path}") # 可以取消注释以获得更详细的输出

        except Exception as e:
            # 捕获保存单个文件时可能发生的错误
            print(f"  错误：保存文件 {output_csv_filename} 时出错: {e}")
            # 可以选择是跳过这个文件继续 (continue) 还是停止 (break 或不处理让循环结束)
            # continue

    print(f"\n已完成为年份 {target_year} 生成所有 12 个月的 CSV 文件。")

else:
    # 如果开始就没有找到点，这里再次提醒
    print("由于未在栅格中找到值为 1 的点，没有生成任何 CSV 文件。")