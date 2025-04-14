import os
import re
from collections import Counter

# --- 配置 ---
# 要检查的输入目录 (包含 .nc4 文件的目录)
input_dir = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\OCO2_GEOS_XCO2同化数据\XCO2_2018"

# --- 脚本执行 ---

print(f"开始检查目录: {input_dir}")

nc4_files = []
dates_extracted = []
non_2018_files = []
filenames_by_date = {} # 用于存储每个日期对应的文件名列表

# 正则表达式用于从文件名中提取 YYYYMMDD
# 假设文件名格式为 ..._YYYYMMDD_...nc4
date_pattern = re.compile(r'_(\d{8})_') # 匹配下划线包围的8位数字

# 1. 遍历目录，收集 .nc4 文件并提取日期
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".nc4"):
        nc4_files.append(filename)
        
        match = date_pattern.search(filename)
        if match:
            date_str = match.group(1) # 提取8位日期字符串 YYYYMMDD
            year_str = date_str[:4] # 提取年份 YYYY
            
            if year_str == "2018":
                dates_extracted.append(date_str)
                # 记录日期对应的文件名
                if date_str not in filenames_by_date:
                    filenames_by_date[date_str] = []
                filenames_by_date[date_str].append(filename)
            else:
                non_2018_files.append(filename)
        else:
            print(f"  警告：无法从文件名 '{filename}' 中提取日期。")

# 2. 分析结果
total_nc4_files = len(nc4_files)
unique_2018_dates = set(dates_extracted)
count_unique_2018_dates = len(unique_2018_dates)

# 使用 Counter 查找重复日期
date_counts = Counter(dates_extracted)
duplicate_dates = {date: count for date, count in date_counts.items() if count > 1}

# 3. 打印报告
print("\n--- 检查报告 ---")
print(f"在目录中共找到 {total_nc4_files} 个 .nc4 文件。")

if non_2018_files:
    print(f"\n发现 {len(non_2018_files)} 个非 2018 年的文件:")
    for fname in non_2018_files:
        print(f"  - {fname}")
else:
    print("\n所有 .nc4 文件名中的年份均为 2018。")

print(f"\n共提取到 {count_unique_2018_dates} 个不同的 2018 年日期。")

if duplicate_dates:
    print(f"\n发现 {len(duplicate_dates)} 个日期存在重复文件:")
    for date, count in duplicate_dates.items():
        print(f"  - 日期 {date} 出现了 {count} 次。涉及文件:")
        # 打印出具体是哪些文件重复了
        if date in filenames_by_date:
             for fname in filenames_by_date[date]:
                  print(f"    * {fname}")
else:
    print("\n未发现重复日期的文件。")

# 4. 最终判断
print("\n--- 结论 ---")
if total_nc4_files == 366 and count_unique_2018_dates == 365 and len(duplicate_dates) == 1 and not non_2018_files:
     dup_date = list(duplicate_dates.keys())[0]
     print(f"原因分析：目录中总文件数为 366，包含 365 个独立日期，")
     print(f"问题在于日期 '{dup_date}' 存在重复文件，导致总数增加。")
elif total_nc4_files == 366 and non_2018_files:
     print(f"原因分析：目录中总文件数为 366，但包含 {len(non_2018_files)} 个非 2018 年的文件。")
     print(f"移除这些非 2018 文件后，实际 2018 年的文件数可能是 {total_nc4_files - len(non_2018_files)} 个。")
elif total_nc4_files == 366 and count_unique_2018_dates < 365 :
     print(f"原因分析：目录中总文件数为 366，但 2018 年的独立日期少于 365 天({count_unique_2018_dates}天)。")
     print(f"这表明可能同时存在日期缺失和日期重复的情况，或者包含了非2018年的文件。请检查上面的详细报告。")
elif total_nc4_files != 366:
     print(f"原因分析：目录中的 .nc4 文件总数实际上是 {total_nc4_files} 个，并非 366 个。之前的统计可能有误。")
else:
     print("原因分析：情况比较复杂，请仔细核对上面的文件列表、非2018文件列表和重复日期列表。")

print("\n检查完成。")