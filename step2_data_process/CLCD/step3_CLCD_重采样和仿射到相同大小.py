# step3_CLCD_重采样和仿射到相同大小

import rasterio
from rasterio.warp import reproject, Resampling

from rasterio import mask # 导入 rasterio.mask
import os
from rasterio import Affine
import numpy as np

def crop_raster_by_extent(input_path, reference_path, output_path):
    """
    根据栅格的外接矩形裁剪栅格 (不扩展范围，地理坐标裁剪，尽量不移动像素位置).

    Args:
        input_path (str): 输入栅格文件路径。
        output_path (str): 输出裁剪后栅格文件路径。
    """
    try:

        with rasterio.open(reference_path) as reference:
            bounds = reference.bounds

            # 直接使用原始边界作为裁剪边界，不进行扩展
            crop_bounds = (
                    bounds.left,
                    bounds.bottom,
                    bounds.right,
                    bounds.top
                )

            # 构建裁剪形状 (GeoJSON 格式的 Polygon)
            crop_shape = {
                    "type": "Polygon",
                    "coordinates": [[
                        [crop_bounds[0], crop_bounds[3]], # 左上角
                        [crop_bounds[2], crop_bounds[3]], # 右上角
                        [crop_bounds[2], crop_bounds[1]], # 右下角
                        [crop_bounds[0], crop_bounds[1]], # 左下角
                        [crop_bounds[0], crop_bounds[3]]  # 回到左上角闭合
            ]]
            }
  
            with rasterio.open(input_path) as src:

                # 使用 rasterio.mask.mask 进行裁剪
                out_image, out_transform = mask.mask(src, [crop_shape], crop=True)
                out_meta = src.meta.copy()

                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1], # 注意 shape 的顺序 (bands, height, width)
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "bigtiff": "YES"  
                })

                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)

                print(f"已裁剪栅格 (地理坐标裁剪, 无扩展): {output_path}")
                return True

    except Exception as e:
        print(f"裁剪文件 {input_path} 时发生错误: {str(e)}")
        return False



def align_raster(input_path, reference_path, output_path_aligned, output_path_cropped):
    try:
        # 先裁剪输入栅格
        if not crop_raster_by_extent(input_path, reference_path, output_path_cropped):
            return False

        # 读取参考栅格获取目标参数
        with rasterio.open(reference_path) as src2:
            dst_crs = src2.crs
            dst_transform = src2.transform
            dst_width = src2.width
            dst_height = src2.height

        # 对裁剪后的输入栅格进行重投影和重采样
        with rasterio.open(output_path_cropped) as src1:
            meta = src1.meta.copy()
            meta.update({
                'crs': dst_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height,  # 确保设置高度
                'bigtiff': 'YES'
            })

            # 创建输出文件并进行重投影
            with rasterio.open(output_path_aligned, 'w', **meta) as dst1:
                for i in range(1, src1.count + 1):
                    reproject(
                        source=rasterio.band(src1, i),
                        destination=rasterio.band(dst1, i),
                        src_transform=src1.transform,
                        src_crs=src1.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear
                    )
        print(f"已保存对齐后的栅格: {output_path_aligned}")
        return True

    except Exception as e:
        print(f"处理文件 {input_path} 时发生错误: {str(e)}")
        return False
    
def main():
    # 定义输入输出路径
    input_dir = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\CLCD\原始数据_众数重采样_setnull'
    output_dir = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\CLCD\原始数据_众数重采样_setnull_仿射'
    reference_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\范围数据\标准栅格.tif'
    cropped_subdir = 'cropped' # 用于存放裁剪后栅格的子目录

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
        output_path_aligned = os.path.join(output_dir, tif_file)
        output_path_cropped = os.path.join(cropped_output_dir, tif_file) # 裁剪后的文件路径

        print(f"\n处理第 {i}/{total_files} 个文件: {tif_file}")

        if align_raster(input_path, reference_path, output_path_aligned, output_path_cropped):
            successful += 1
        else:
            failed += 1

    # 打印处理结果统计
    print("\n处理完成:")
    print(f"成功处理: {successful} 个文件")
    print(f"处理失败: {failed} 个文件")
    print(f"总计文件: {total_files} 个")

    # 验证最后一个处理的文件
    if successful > 0:
        print("\n验证最后处理的文件:")
        with rasterio.open(output_path_aligned) as aligned1, rasterio.open(reference_path) as src2:
            print(f"参考栅格大小: {src2.width} x {src2.height}")
            print(f"对齐后栅格大小: {aligned1.width} x {aligned1.height}")
            print(f"参考栅格变换矩阵: {src2.transform}")
            print(f"对齐后变换矩阵: {aligned1.transform}")

            if (aligned1.transform == src2.transform and
                aligned1.width == src2.width and
                aligned1.height == src2.height):
                print("栅格已成功对齐到参考栅格的网格和分辨率")
            else:
                print("对齐结果有误，请检查")

if __name__ == "__main__":
    main()
