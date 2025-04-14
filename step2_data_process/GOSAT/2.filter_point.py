import pandas as pd

# 输入和输出文件路径
input_csv_path = r'E:\地理所\工作\徐州卫星同化_2025.2\gosat\GOSAT2_XCO2_Data.csv'
output_csv_path = r'E:\地理所\工作\徐州卫星同化_2025.2\gosat\GOSAT2_XCO2_Data_filtered_xuzhou.csv'

# 经纬度范围
min_latitude = 33
max_latitude = 36
min_longitude = 116
max_longitude = 119

# 加载数据
print("Loading data...")
data = pd.read_csv(input_csv_path)

# 筛选符合条件的观测点
print("Filtering data...")
filtered_data = data[
    (data["Latitude"] >= min_latitude) &
    (data["Latitude"] <= max_latitude) &
    (data["Longitude"] >= min_longitude) &
    (data["Longitude"] <= max_longitude)
]

# 保存筛选后的数据到CSV文件
print(f"Saving filtered data to {output_csv_path}...")
filtered_data.to_csv(output_csv_path, index=False)

print("Filtering completed!")
