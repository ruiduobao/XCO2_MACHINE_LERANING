import rasterio
import os
import glob
import warnings

# 忽略某些不影响维度检查的警告 (可选)
# warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# 1. 定义栅格文件根目录 和 子文件夹
root_dir = r'E:\地理所\论文\中国XCO2论文_2025.04\数据' # 使用原始字符串路径
subfolders_to_scan = [
    '纬度栅格', '经度栅格', '每月时间戳的栅格数据', '坡向数据',
    '坡度数据', 'DEM', '夜光遥感', 'ERA5', '气溶胶厚度',
    'carbon_tracer', 'landscan', 'odiac', '人类足迹数据',
    'OCO2_GEOS_XCO2同化数据', 'CAMS', 'CLCD', 'modis_landcover', 'NDVI'
]

# --- 步骤 2: 扫描子文件夹查找所有 .tif 文件 ---
all_tif_files = []
print(f"正在根目录 '{root_dir}' 下扫描指定的子文件夹...")

for folder_name in subfolders_to_scan:
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(folder_path):
        print(f"  警告：子文件夹 '{folder_path}' 不存在，跳过。")
        continue

    print(f"  扫描文件夹: '{folder_path}'")
    # 使用 glob 查找当前文件夹下所有的 .tif 文件
    # 确保同时匹配 .tif 和 .TIF (大小写不敏感，但在某些系统上 glob 可能敏感)
    found_files = glob.glob(os.path.join(folder_path, '*.tif'))
    found_files.extend(glob.glob(os.path.join(folder_path, '*.TIF')))
    # 去重，以防万一
    found_files = list(set(found_files))

    # 排除您之前脚本中可能生成的临时文件 (如果需要)
    # found_files = [f for f in found_files if '_with_crs.tif' not in os.path.basename(f)]

    all_tif_files.extend(found_files)
    print(f"    找到 {len(found_files)} 个 .tif 文件。")


if not all_tif_files:
    print("\n错误：在指定的子文件夹中未找到任何 .tif 文件。请检查 root_dir 和 subfolders_to_scan 是否正确。")
    exit() # 如果没有找到文件则退出

print(f"\n总共找到 {len(all_tif_files)} 个 .tif 文件需要检查维度。")
print("-" * 40)


# --- 步骤 3: 遍历找到的TIFF文件检查维度 ---
reference_shape = None         # 用于存储参考维度 (height, width)
reference_file_path = None     # 用于存储设置参考维度的文件路径
dimension_mismatch_files = {}  # 字典用于存储路径和不匹配的维度 {path: (height, width)}
failed_files = []              # 列表用于存储无法处理的文件路径及原因

for tif_path in sorted(all_tif_files): # 排序以确保一致的参考文件选择
    print(f"正在检查: {tif_path}")
    try:
        # 使用 with 语句确保文件正确关闭
        with rasterio.open(tif_path) as src:
            # 获取当前文件的维度 (height=行数, width=列数)
            current_shape = (src.height, src.width)

            if reference_shape is None:
                # 这是第一个成功打开的文件，将其维度设为参考标准
                reference_shape = current_shape
                reference_file_path = tif_path
                print(f"  信息：将此文件维度设为参考标准: {reference_shape} (行数, 列数)")
            elif current_shape != reference_shape:
                # 当前文件维度与参考标准不同
                print(f"  *** 警告：维度不匹配! 文件维度: {current_shape}, 参考维度: {reference_shape}")
                dimension_mismatch_files[tif_path] = current_shape # 记录不匹配的文件及其维度
            else:
                # 维度匹配
                print(f"  信息：维度匹配 ({current_shape})")

    except rasterio.RasterioIOError as e:
        error_msg = f"无法打开或读取文件: {e}"
        print(f"  错误：{error_msg}")
        failed_files.append(f"{tif_path} ({error_msg})")
    except Exception as e:
        error_msg = f"检查文件时发生未知错误: {e}"
        print(f"  错误：{error_msg}")
        failed_files.append(f"{tif_path} ({error_msg})")
    finally:
        print("-" * 30) # 在每个文件处理后打印分隔符

# --- 步骤 4: 打印总结报告 ---
print("\n" + "=" * 20 + " 维度检查结果总结 " + "=" * 20)

if reference_file_path:
    print(f"参考维度 (行数, 列数): {reference_shape}")
    print(f"参考文件路径: {reference_file_path}")
else:
    # 如果所有文件都读取失败，reference_shape 会是 None
    print("未能从任何文件中成功获取参考维度（可能所有文件都无法读取）。")

if dimension_mismatch_files:
    print(f"\n发现 {len(dimension_mismatch_files)} 个文件维度与参考标准不匹配:")
    # 对字典按文件名排序后输出，更清晰
    for f_path in sorted(dimension_mismatch_files.keys()):
        shape = dimension_mismatch_files[f_path]
        print(f"- 文件: {f_path}")
        print(f"    维度: {shape} (行数, 列数)")
else:
    if reference_file_path: # 只有在有参考标准的情况下，"都匹配"才有意义
       print("\n所有成功检查的文件维度均与参考标准相同。")

if failed_files:
    print(f"\n处理失败或无法读取的文件 ({len(failed_files)}):")
    for f_path_err in sorted(failed_files):
        print(f"- {f_path_err}")
else:
    print("\n所有文件均成功读取并完成维度检查。")

print("\n脚本执行完毕。")