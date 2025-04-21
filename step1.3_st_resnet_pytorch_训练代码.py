import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from collections import OrderedDict

# ===== 添加中文字体支持 =====
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适用于无显示器环境
# 设置中文字体
import matplotlib.font_manager as fm
# 查找系统中的中文字体（以下是常见的中文字体，二选一即可）
# 微软雅黑
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体为微软雅黑
    # 测试字体是否可用
    matplotlib.font_manager.findfont('Microsoft YaHei')
    print("已设置字体：Microsoft YaHei (微软雅黑)")
except:
    try:
        # SimHei 黑体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
        matplotlib.font_manager.findfont('SimHei')
        print("已设置字体：SimHei (黑体)")
    except:
        print("警告：未找到中文字体，图表中文可能显示为方块")

plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
# ===== 中文字体设置结束 =====

# --- 配置 ---
# TIF文件目录
ROOT_DIR = r"E:\地理所\论文\中国XCO2论文_2025.04\数据"
# ... existing code ... 