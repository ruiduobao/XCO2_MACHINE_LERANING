import rasterio
import numpy as np
import os

# --- 输入和输出文件路径 ---
# 输入文件是上一步生成的删除了波段的文件
input_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\modis_landcover\原始数据_删掉某个波段\MODIS_LANDCOVER2018_subset.tif'
# 创建一个新的输出文件名
output_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\modis_landcover\原始数据_删掉某个波段\MODIS_LANDCOVER2018_subset_int16.tif'

# --- 值替换和目标 NoData 值 ---
value_to_replace = 0
replacement_value = -9999
target_nodata = -9999
target_dtype = rasterio.int16 # 目标数据类型

print(f"正在处理输入文件: {input_path}")
print(f"将数据类型转为: {target_dtype}")
print(f"将像素值 {value_to_replace} 替换为: {replacement_value}")
print(f"设置输出 NoData 值为: {target_nodata}")
print(f"输出文件将保存为: {output_path}")

try:
    # --- 1. 打开源数据集并读取元数据 ---
    with rasterio.open(input_path) as src:
        # 获取源文件的 profile
        profile = src.profile

        profile.update(
        nodata=-9999,
        compress='lzw',  # 添加压缩
        tiled=True,      # 使用分块
        blockxsize=256,  # 设置块大小
        blockysize=256,
        BIGTIFF='YES'
        )
        
        print(f"源文件数据类型: {profile['dtype']}, 波段数: {src.count}")

        # --- 2. 更新输出文件的 profile ---
        profile['dtype'] = target_dtype # 设置目标数据类型
        profile['nodata'] = target_nodata # 设置目标 NoData 值
        # 其他元数据（CRS, transform, compression, tiled等）会自动保留

        # 检查并移除 profile 中的 colorinterp，因为它不能通过 set_colorinterp 设置
        if 'colorinterp' in profile:
             del profile['colorinterp']
             print("从输出 profile 中移除 colorinterp 以避免冲突。")


        print(f"输出文件 profile 更新: dtype={profile['dtype']}, nodata={profile['nodata']}")

        # --- 3. 创建输出文件并写入处理后的数据 ---
        with rasterio.open(output_path, 'w', **profile) as dst:
            # 逐个波段处理
            for band_index in range(1, src.count + 1):
                print(f"  正在处理波段 {band_index}/{src.count}...")
                # 读取当前波段数据
                data_uint8 = src.read(band_index) # 原始 uint8 数据

                # --- !!! 修改顺序：先转换类型 !!! ---
                print(f"    将数据转换为 {target_dtype}...")
                data_int16 = data_uint8.astype(target_dtype)
                # --- !!! 修改顺序结束 !!! ---

                # --- !!! 修改顺序：后替换值 !!! ---
                # 在 int16 数据上进行替换，避免 uint8 溢出警告
                print(f"    查找值为 {value_to_replace} 的像素...")
                mask_to_replace = (data_int16 == value_to_replace)
                num_replaced = np.sum(mask_to_replace)
                print(f"    找到并替换 {num_replaced} 个像素。")
                data_int16[mask_to_replace] = replacement_value
                # --- !!! 修改顺序结束 !!! ---

                # 写入处理后的 int16 数据到目标文件
                print(f"    写入波段 {band_index} 到输出文件...")
                dst.write(data_int16, band_index)

                # 可选：复制波段元数据（描述、标签）
                try:
                    tags = src.tags(band_index)
                    if tags:
                        dst.update_tags(band_index, **tags)
                    # --- !!! 移除 set_colorinterp !!! ---
                    # ci = src.colorinterp[band_index-1]
                    # dst.set_colorinterp(band_index, ci) # 移除此行
                    # --- !!! 移除结束 !!! ---
                    desc = src.descriptions[band_index-1]
                    if desc:
                        dst.set_band_description(band_index, desc)
                except IndexError:
                    pass # 忽略元数据复制中的小错误
                except Exception as e_meta:
                     print(f"    警告：复制波段 {band_index} 的元数据时出错: {e_meta}")


    print("\n处理完成！")
    print(f"已创建新文件: {output_path}")

except FileNotFoundError:
    print(f"错误：输入文件未找到 {input_path}")
except Exception as e:
    print(f"处理过程中发生错误: {e}")