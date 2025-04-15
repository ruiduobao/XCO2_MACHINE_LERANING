import os
import pandas as pd

# 指定文件夹路径
folder_path = r'E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\卫星数据\gosat'

# 遍历文件夹下的所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):  # 确保只处理CSV文件
        file_path = os.path.join(folder_path, filename)
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 检查"time"列是否存在
        if 'time' in df.columns:
            # 将"time"列转换为datetime对象，指定时间格式
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S.%f')
            
            # 提取年、月、日、时、分、秒，并添加或更新对应列
            df['year'] = df['time'].dt.year
            df['month'] = df['time'].dt.month
            df['day'] = df['time'].dt.day
            df['hour'] = df['time'].dt.hour
            df['minute'] = df['time'].dt.minute
            df['second'] = df['time'].dt.second
            
            # 保存处理后的数据回原文件
            df.to_csv(file_path, index=False)
            print(f'已处理文件：{filename}')
        else:
            print(f'文件 {filename} 中未找到"time"列，跳过处理')