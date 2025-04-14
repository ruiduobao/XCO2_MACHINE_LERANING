import os
from datetime import datetime, timedelta

# 定义路径
path = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\OCO2"

# 定义日期范围（这里假设从2018年1月1日到今天，2025年4月10日）
start_date = datetime(2018, 1, 1)
end_date = datetime(2018, 12, 31)  # 当前日期

# 获取路径下所有文件夹名称
existing_folders = set(os.listdir(path))

# 生成所有应该存在的日期文件夹名称
all_dates = set()
current_date = start_date
while current_date <= end_date:
    folder_name = current_date.strftime("%Y%m%d")  # 格式化为 YYYYMMDD
    all_dates.add(folder_name)
    current_date += timedelta(days=1)

# 找出缺失的文件夹
missing_folders = all_dates - existing_folders

# 按日期排序并输出结果
missing_folders = sorted(list(missing_folders))
if missing_folders:
    print("以下日期的文件夹缺失：")
    for folder in missing_folders:
        print(folder)
else:
    print("没有缺失的文件夹。")

# 输出缺失总数
print(f"总计缺失 {len(missing_folders)} 个文件夹。")