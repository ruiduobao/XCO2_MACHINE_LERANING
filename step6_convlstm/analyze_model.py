import torch
import os
import sys

# 添加父目录到路径以导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型模块
from step6_convlstm.model import XCO2ConvLSTM

# 设置模型路径
model_path = r"E:\地理所\论文\中国XCO2论文_2025.04\代码\step6_convlstm\已训练模型\best_xco2_convlstm_model.pth"

# 加载模型状态字典
state_dict = torch.load(model_path)

# 打印所有权重
for key in state_dict.keys():
    print(key)

# 查看特定层的详细信息
if "encoder.cell_list.0.conv.weight" in state_dict:
    weight = state_dict["encoder.cell_list.0.conv.weight"]
    print(f"Shape: {weight.shape}")
    print(f"Type: {weight.dtype}")
    print(f"Device: {weight.device}")



# debug_aux_data.py
import os
import sys
import numpy as np
from step6_convlstm.data_loader import list_xco2_files, create_auxiliary_input

# 设置路径
data_dir = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\XGBOOST_XCO2'
xco2_files = list_xco2_files(data_dir)

# 设置辅助数据目录
def setup_auxiliary_data_dirs():
    root_dir = r'E:\地理所\论文\中国XCO2论文_2025.04\数据'
    aux_dirs = {
        'Lantitude': os.path.join(root_dir, '纬度栅格'),
        'Longtitude': os.path.join(root_dir, '经度栅格'),
        'ERA5Land': os.path.join(root_dir, 'ERA5'),
        'CT2019B': os.path.join(root_dir, 'carbon_tracer'),
        'landscan': os.path.join(root_dir, 'landscan'),
        'OCO2GEOS': os.path.join(root_dir, 'OCO2_GEOS_XCO2同化数据'),
        'CAMStcco2': os.path.join(root_dir, 'CAMS'),
        'MODISLANDCOVER': os.path.join(root_dir, 'modis_landcover'),
    }
    return aux_dirs

# 获取辅助数据目录
aux_dirs = setup_auxiliary_data_dirs()

# 测试第一个文件
if xco2_files:
    first_file = xco2_files[0]
    print(f"正在加载文件: {first_file}")
    
    # 加载辅助数据
    aux_data = create_auxiliary_input(first_file, aux_dirs)
    
    # 打印形状和通道数
    print(f"辅助数据形状: {aux_data.shape}")
    print(f"辅助数据通道数: {aux_data.shape[0]}")
    
    # 打印每个特征目录内的文件数量
    for name, dir_path in aux_dirs.items():
        files = os.listdir(dir_path)
        print(f"{name}: {len(files)} 文件")