# step2_CAMS_重采样和仿射到相同大小
# 使用最临近采样
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio import mask # 导入 rasterio.mask
import os
from rasterio import Affine
import numpy as np

# -------------------------------------------------------------
# 修改后的 crop_raster_by_extent 函数
# -------------------------------------------------------------
def crop_raster_by_extent(input_path, reference_path, output_path, buffer_degrees=0.0):
    """
    根据参考栅格的外接矩形裁剪输入栅格，并可选择向外扩展范围。

    Args:
        input_path (str): 输入栅格文件路径 (例如 CAMS)。
        reference_path (str): 参考栅格文件路径 (定义裁剪范围)。
        output_path (str): 输出裁剪后栅格文件路径。
        buffer_degrees (float): 向外扩展的度数。默认为 0.0 (不扩展)。
                                对于 0.75° 的 CAMS 数据，建议设置为 0.75。
    """
    try:
        # 打开参考栅格获取边界
        with rasterio.open(reference_path) as reference:
            bounds = reference.bounds
            print(f"参考栅格原始边界: {bounds}")

        # 计算扩展后的边界
        # 注意：rasterio.bounds 是 (left, bottom, right, top)
        crop_left = bounds.left - buffer_degrees
        crop_bottom = bounds.bottom - buffer_degrees
        crop_right = bounds.right + buffer_degrees
        crop_top = bounds.top + buffer_degrees

        extended_bounds = (crop_left, crop_bottom, crop_right, crop_top)
        print(f"扩展 {buffer_degrees}° 后的裁剪边界: {extended_bounds}")

        # 构建裁剪形状 (GeoJSON 格式的 Polygon)
        crop_shape = {
            "type": "Polygon",
            "coordinates": [[
                [extended_bounds[0], extended_bounds[3]], # 左上角
                [extended_bounds[2], extended_bounds[3]], # 右上角
                [extended_bounds[2], extended_bounds[1]], # 右下角
                [extended_bounds[0], extended_bounds[1]], # 左下角
                [extended_bounds[0], extended_bounds[3]]  # 回到左上角闭合
            ]]
        }

        # 打开输入栅格进行裁剪
        with rasterio.open(input_path) as src:
            # 检查输入栅格的CRS是否与参考栅格一致，如果不一致，mask可能需要处理投影问题
            # 但通常地理坐标系（如WGS84）下直接用经纬度裁剪是常见的
            if reference.crs != src.crs:
                 print(f"警告: 输入栅格CRS ({src.crs}) 与参考栅格CRS ({reference.crs}) 不同。裁剪基于参考栅格的地理坐标边界。")

            print(f"正在使用扩展边界裁剪: {input_path}")
            # 使用 rasterio.mask.mask 进行裁剪
            # crop=True 表示输出栅格的范围将紧密贴合裁剪形状
            # all_touched=True 可以考虑包含所有与形状相交的像素，但对于扩展可能不是必需的
            out_image, out_transform = mask.mask(src, [crop_shape], crop=True)
            out_meta = src.meta.copy()

            # 更新元数据以反映裁剪后的尺寸和变换
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1], # 注意 shape 的顺序 (bands, height, width)
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": src.crs, # 保持原始CRS，因为只是裁剪
                "bigtiff": "YES"
            })

            # 写入裁剪后的栅格
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)

            print(f"已保存扩展裁剪后的栅格: {output_path}")
            print(f"裁剪后栅格大小: {out_meta['width']} x {out_meta['height']}")
            print(f"裁剪后变换矩阵: {out_meta['transform']}")
            return True

    except Exception as e:
        print(f"裁剪文件 {input_path} 时发生错误: {str(e)}")
        # 可以在这里加入更详细的错误跟踪，例如 import traceback; traceback.print_exc()
        return False

# -------------------------------------------------------------
# 修改后的 align_raster 函数 (调用扩展裁剪)
# -------------------------------------------------------------
def align_raster(input_path, reference_path, output_path_aligned, output_path_cropped):
    try:
        # --- 修改点：在这里传递 buffer_degrees 参数 ---
        # CAMS 分辨率是 0.75°, 所以扩展 0.75°
        cams_buffer = 0.625
        print(f"步骤 1: 对输入栅格进行扩展裁剪 (buffer={cams_buffer}°)")
        if not crop_raster_by_extent(input_path, reference_path, output_path_cropped, buffer_degrees=cams_buffer):
            print("扩展裁剪失败，跳过此文件。")
            return False
        print("扩展裁剪完成。")

        # 读取参考栅格获取目标参数 (CRS, Transform, Width, Height)
        print("步骤 2: 读取参考栅格参数用于重投影。")
        with rasterio.open(reference_path) as src2:
            dst_crs = src2.crs
            dst_transform = src2.transform
            dst_width = src2.width
            dst_height = src2.height
            print(f"目标参数 - CRS: {dst_crs}, Transform: {dst_transform}, Width: {dst_width}, Height: {dst_height}")

        # 对 扩展裁剪后 的输入栅格进行重投影和重采样
        print(f"步骤 3: 重投影扩展裁剪后的栅格 ({output_path_cropped}) 到目标网格。")
        with rasterio.open(output_path_cropped) as src1:
            # 准备输出文件的元数据
            meta = src1.meta.copy()
            meta.update({
                'crs': dst_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height,
                'nodata': src1.nodata, # 确保传递 nodata 值
                'bigtiff': 'YES'
            })
            # 确保驱动是 GTiff
            meta['driver'] = 'GTiff'


            print(f"源 (裁剪后) - CRS: {src1.crs}, Transform: {src1.transform}, Width: {src1.width}, Height: {src1.height}")
            print(f"目标 (重投影) - CRS: {meta['crs']}, Transform: {meta['transform']}, Width: {meta['width']}, Height: {meta['height']}")

            # 创建输出文件并进行重投影 (使用最邻近插值)
            with rasterio.open(output_path_aligned, 'w', **meta) as dst1:
                print(f"正在执行重投影 (Nearest Neighbor)...")
                for i in range(1, src1.count + 1):
                    reproject(
                        source=rasterio.band(src1, i),
                        destination=rasterio.band(dst1, i),
                        src_transform=src1.transform,
                        src_crs=src1.crs,
                        src_nodata=src1.nodata, # 指定源 nodata
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        dst_nodata=meta.get('nodata'), # 指定目标 nodata
                        resampling=Resampling.nearest # 明确使用最邻近插值
                    )
                print(f"重投影完成。")

        print(f"已保存对齐后的栅格: {output_path_aligned}")
        return True

    except Exception as e:
        print(f"处理文件 {input_path} 时在 align_raster 函数中发生错误: {str(e)}")
        # import traceback; traceback.print_exc() # 取消注释以获取详细堆栈跟踪
        return False

# main 函数保持不变，它会调用修改后的 align_raster
def main():
    # 定义输入输出路径
    input_dir = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\landscan\原始数据'
    output_dir = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\landscan\原始数据_仿射'
    reference_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\范围数据\标准栅格.tif'
    cropped_subdir = 'cropped_extended' # 修改子目录名以反映扩展裁剪

    cropped_output_dir = os.path.join(output_dir, cropped_subdir)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cropped_output_dir, exist_ok=True)


    # 获取所有tif文件
    tif_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    total_files = len(tif_files)

    print(f"共找到 {total_files} 个TIF文件需要处理")

    # 处理每个文件
    successful = 0
    failed = 0

    for i, tif_file in enumerate(tif_files, 1):
        input_path = os.path.join(input_dir, tif_file)
        # 最终对齐输出的路径
        output_path_aligned = os.path.join(output_dir, tif_file)
        # 扩展裁剪后的中间文件路径
        output_path_cropped = os.path.join(cropped_output_dir, tif_file)

        print(f"\n------------------- 开始处理第 {i}/{total_files} 个文件: {tif_file} -------------------")

        # 检查最终输出文件是否已存在，如果存在则跳过 (可选)
        # if os.path.exists(output_path_aligned):
        #     print(f"输出文件 {output_path_aligned} 已存在，跳过。")
        #     successful += 1 # 或者根据需要计数
        #     continue

        if align_raster(input_path, reference_path, output_path_aligned, output_path_cropped):
            successful += 1
        else:
            failed += 1
            print(f"!!! 文件 {tif_file} 处理失败。")
        print(f"------------------- 文件 {tif_file} 处理结束 -------------------\n")

    # 打印处理结果统计
    print("\n=================== 处理完成 ===================")
    print(f"成功处理: {successful} 个文件")
    print(f"处理失败: {failed} 个文件")
    print(f"总计文件: {total_files} 个")
    print(f"对齐后的栅格保存在: {output_dir}")
    print(f"扩展裁剪后的中间文件保存在: {cropped_output_dir}")
    print("==============================================")

    # 验证最后一个成功处理的文件 (如果存在)
    if successful > 0 and 'output_path_aligned' in locals(): # 确保变量存在
        print("\n验证最后处理的文件:")
        try:
             with rasterio.open(output_path_aligned) as aligned1, rasterio.open(reference_path) as src2:
                print(f"参考栅格 ('{os.path.basename(reference_path)}')")
                print(f"  - 大小: {src2.width} x {src2.height}")
                print(f"  - 变换矩阵 (Transform): {src2.transform}")
                print(f"  - 坐标参考系 (CRS): {src2.crs}")

                print(f"对齐后栅格 ('{os.path.basename(output_path_aligned)}')")
                print(f"  - 大小: {aligned1.width} x {aligned1.height}")
                print(f"  - 变换矩阵 (Transform): {aligned1.transform}")
                print(f"  - 坐标参考系 (CRS): {aligned1.crs}")

                # 进行更严格的比较，考虑浮点数精度问题
                transform_match = np.allclose(np.array(aligned1.transform), np.array(src2.transform), atol=1e-8) # 比较仿射变换矩阵
                width_match = aligned1.width == src2.width
                height_match = aligned1.height == src2.height
                crs_match = aligned1.crs == src2.crs # 比较 CRS

                if transform_match and width_match and height_match and crs_match:
                    print("\n[验证成功] 栅格已成功对齐到参考栅格的网格、分辨率和CRS。")
                else:
                    print("\n[验证失败] 对齐结果与参考栅格不完全匹配，请检查：")
                    if not transform_match: print("  - Transform 不匹配")
                    if not width_match: print("  - Width 不匹配")
                    if not height_match: print("  - Height 不匹配")
                    if not crs_match: print("  - CRS 不匹配")
        except Exception as e:
             print(f"验证文件时出错: {e}")

if __name__ == "__main__":
    main()