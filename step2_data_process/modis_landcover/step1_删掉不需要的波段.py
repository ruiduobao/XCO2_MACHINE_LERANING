# step1_删掉不需要的波段
import rasterio
import os
import numpy as np

# --- 输入和输出文件路径 ---
input_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\modis_landcover\原始数据\MODIS_LANDCOVER2018.tif'
output_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\modis_landcover\原始数据\MODIS_LANDCOVER2018_subset.tif' # 输出文件名加后缀

# --- 要删除的波段号 (基于 1 的索引) ---
bands_to_remove = [12, 13]

print(f"正在处理输入文件: {input_path}")
print(f"将要删除的波段: {bands_to_remove}")
print(f"输出文件将保存为: {output_path}")

try:
    # --- 1. 打开源数据集并读取元数据 ---
    with rasterio.open(input_path) as src:
        # 获取源文件的 profile (包含所有元数据)
        profile = src.profile
        print(f"源文件共有 {src.count} 个波段。")

        # 检查要删除的波段是否存在
        if any(b > src.count or b <= 0 for b in bands_to_remove):
            print(f"错误：要删除的波段号 {bands_to_remove} 超出源文件波段范围 (1-{src.count})。")
            exit()

        # --- 2. 确定要保留的波段索引 ---
        # 创建一个从 1 到 n 的列表，然后移除要删除的波段
        bands_to_keep_indices = [b for b in range(1, src.count + 1) if b not in bands_to_remove]

        if not bands_to_keep_indices:
            print("错误：删除指定波段后没有剩余波段了。")
            exit()

        print(f"将要保留的波段号: {bands_to_keep_indices}")
        output_band_count = len(bands_to_keep_indices)
        print(f"输出文件将有 {output_band_count} 个波段。")

        # --- 3. 更新输出文件的 profile ---
        profile['count'] = output_band_count

        # 处理 NoData 值：如果源文件为每个波段定义了不同的 NoData 值，
        # 我们需要创建一个只包含保留波段的 NoData 值元组。
        if src.nodatavals:
            original_nodatavals = src.nodatavals
            # 创建新的 NoData 元组，只包含保留波段的值
            # 注意：波段索引是 1-based, 元组索引是 0-based
            new_nodatavals = tuple(original_nodatavals[i - 1] for i in bands_to_keep_indices)
            # 检查是否所有保留的 NoData 值都相同，如果是，可以用单个值
            if len(set(new_nodatavals)) == 1:
                 profile['nodata'] = new_nodatavals[0]
            else:
                 profile['nodata'] = new_nodatavals # 保持为元组
            print(f"更新后的 NoData 值: {profile['nodata']}")
        elif profile.get('nodata') is not None:
            # 如果 profile 中有一个单一的 nodata 值，则无需更改
            print(f"保留原始的单一 NoData 值: {profile['nodata']}")
        else:
            print("源文件未设置 NoData 值。")


        # --- 4. 创建输出文件并写入数据 ---
        with rasterio.open(output_path, 'w', **profile) as dst:
            output_band_index = 1 # 输出文件的波段索引从 1 开始
            for input_band_index in bands_to_keep_indices:
                print(f"  正在复制波段 {input_band_index} (源) -> 波段 {output_band_index} (目标)...")
                # 从源文件读取数据
                data = src.read(input_band_index)
                # 写入到目标文件
                dst.write(data, output_band_index)

                # 可选：复制波段描述、颜色解释等元数据 (如果存在)
                try:
                    tags = src.tags(input_band_index)
                    if tags:
                        dst.update_tags(output_band_index, **tags)
                    ci = src.colorinterp[input_band_index-1] # colorinterp 是 0-based list
                    dst.set_colorinterp(output_band_index, ci)
                    desc = src.descriptions[input_band_index-1] # descriptions 是 0-based tuple
                    if desc:
                        dst.set_band_description(output_band_index, desc)
                except IndexError:
                    # 有些旧文件可能没有完整的元数据
                    pass
                except Exception as e_meta:
                    print(f"    警告：复制波段 {input_band_index} 的元数据时出错: {e_meta}")


                output_band_index += 1

    print("\n处理完成！")
    print(f"已创建包含波段 {bands_to_keep_indices} 的新文件: {output_path}")

except FileNotFoundError:
    print(f"错误：输入文件未找到 {input_path}")
except Exception as e:
    print(f"处理过程中发生错误: {e}")