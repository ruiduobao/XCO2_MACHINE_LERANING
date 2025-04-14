import rasterio
import numpy as np
from scipy import ndimage # 用于填充孔洞
import os

# --- 输入和输出路径 ---
input_raster_path = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\矢量数据\中国区域矢量\China_rasterized_0.05deg.tif"
# 创建一个新的输出文件名，避免覆盖原始文件
output_raster_path = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\矢量数据\中国区域矢量\China_rasterized_0.05deg_filled.tif"

# --- 参数 ---
# 从输入文件名或栅格元数据中获取 NoData 和有效值
# 这里我们根据之前的步骤假设 NoData 是 -9999，有效值是 1
no_data_value = -9999
fill_value = 1 # 要填充空洞的值

# --- 开始处理 ---
print(f"开始处理栅格文件: {input_raster_path}")

try:
    # --- 1. 读取栅格数据和元数据 ---
    with rasterio.open(input_raster_path) as src:
        # 读取第一个波段的数据到 numpy 数组
        data_array = src.read(1)
        # 获取原始文件的元数据（包括 CRS, transform, nodata, compression 等）
        profile = src.profile
        # 确保 profile 中的 nodata 值是我们预期的
        if profile.get('nodata') != no_data_value:
            print(f"警告: 文件元数据中的 NoData 值 ({profile.get('nodata')}) 与预期值 ({no_data_value}) 不同。将使用预期值 {no_data_value}。")
            # 如果需要，可以更新 profile 中的 nodata 值
            # profile['nodata'] = no_data_value

        print(f"成功读取栅格数据，尺寸: {data_array.shape}")
        print(f"使用的 NoData 值: {no_data_value}")
        print(f"将用此值填充空洞: {fill_value}")

        # --- 2. 创建二值掩模 ---
        # True 表示有效数据，False 表示 NoData
        binary_mask = (data_array != no_data_value)
        print("创建二值掩模完成。")

        # --- 3. 填充二值掩模中的孔洞 ---
        # ndimage.binary_fill_holes 会填充被 True 包围的 False 区域
        filled_mask = ndimage.binary_fill_holes(binary_mask)
        print("填充二值掩模中的孔洞完成。")

        # --- 4. 更新原始数据数组 ---
        # 找到那些在原始数据中是 NoData，但在填充后掩模中是 True 的像素
        # 这些就是需要被填充的内部空洞
        holes_to_fill = (filled_mask == True) & (data_array == no_data_value)
        num_holes_filled = np.sum(holes_to_fill)
        print(f"识别到并准备填充 {num_holes_filled} 个像素的内部空洞。")

        # 使用 fill_value 更新数据数组
        data_array[holes_to_fill] = fill_value
        print("数据数组更新完成。")

        # --- 5. 写入新的栅格文件 ---
        print(f"准备将结果写入新文件: {output_raster_path}")
        # 使用原始文件的 profile 来创建新文件，确保地理参考等信息一致
        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(data_array, 1) # 将修改后的数组写入第一个波段

        print(f"成功！填充空洞后的栅格已保存至: {output_raster_path}")

except FileNotFoundError:
    print(f"错误：输入文件未找到: {input_raster_path}")
except Exception as e:
    print(f"处理过程中发生错误: {e}")