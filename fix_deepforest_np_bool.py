#!/usr/bin/env python
"""
此脚本用于修复DeepForest库中的numpy.bool弃用问题
"""
import os
import re

# 查找并替换DeepForest库中的文件
def find_and_replace_np_bool():
    # 获取DeepForest库的路径
    import deepforest
    deepforest_path = os.path.dirname(deepforest.__file__)
    print(f"DeepForest库路径: {deepforest_path}")
    
    # 修复forest.py文件
    forest_path = os.path.join(deepforest_path, "forest.py")
    if os.path.exists(forest_path):
        print(f"修复文件: {forest_path}")
        with open(forest_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 将np.bool替换为np.bool_
        modified_content = re.sub(r'np\.bool(?!\w)', 'np.bool_', content)
        
        # 备份原文件
        backup_path = forest_path + '.bak'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"原文件已备份为: {backup_path}")
        
        # 写入修改后的内容
        with open(forest_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print(f"文件已修复")
    
    # 修复_cutils.pyx文件
    cutils_path = os.path.join(deepforest_path, "_cutils.pyx")
    if os.path.exists(cutils_path):
        print(f"修复文件: {cutils_path}")
        with open(cutils_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 将np.bool替换为np.bool_
        modified_content = re.sub(r'np\.bool(?!\w)', 'np.bool_', content)
        
        # 备份原文件
        backup_path = cutils_path + '.bak'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"原文件已备份为: {backup_path}")
        
        # 写入修改后的内容
        with open(cutils_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print(f"文件已修复")
        
        print("\n警告: _cutils.pyx是Cython源文件，修改后可能需要重新编译才能生效")
        print("如果这个修复不起作用，您可能需要重新安装DeepForest或使用较低版本的NumPy")
    
    return True

if __name__ == "__main__":
    print("开始修复DeepForest库中的numpy.bool弃用问题...")
    success = find_and_replace_np_bool()
    if success:
        print("\n修复完成！请尝试重新运行您的DeepForest代码")
    else:
        print("\n修复失败，请考虑降级NumPy版本或联系DeepForest开发人员") 