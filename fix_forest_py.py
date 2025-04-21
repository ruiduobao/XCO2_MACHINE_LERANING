#!/usr/bin/env python
"""
此脚本专门修复DeepForest库中forest.py文件的np.bool问题
"""
import os

# 要替换的文本
SEARCH_TEXT = "sample_mask = _LIB._c_sample_mask(sample_indices, n_samples)"
REPLACE_TEXT = """# Fix for NumPy bool deprecation
try:
    sample_mask = _LIB._c_sample_mask(sample_indices, n_samples)
except AttributeError as e:
    if "module 'numpy' has no attribute 'bool'" in str(e):
        # Monkey patch the numpy bool issue
        import numpy as np
        np.bool = bool  # Temporary fix during function execution
        sample_mask = _LIB._c_sample_mask(sample_indices, n_samples)
    else:
        raise"""

def fix_forest_py():
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
        
        # 检查代码是否已经被修复
        if REPLACE_TEXT in content:
            print("此文件已经被修复，无需再次修改")
            return True
        
        # 修改代码
        if SEARCH_TEXT in content:
            modified_content = content.replace(SEARCH_TEXT, REPLACE_TEXT)
            
            # 备份原文件
            backup_path = forest_path + '.bak'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"原文件已备份为: {backup_path}")
            
            # 写入修改后的内容
            with open(forest_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"文件已修复")
            return True
        else:
            print(f"未在文件中找到需要替换的代码: {SEARCH_TEXT}")
            print("请检查DeepForest版本是否与预期一致，或者联系开发人员")
            return False
    else:
        print(f"未找到文件: {forest_path}")
        return False

if __name__ == "__main__":
    print("开始修复DeepForest库的forest.py文件...")
    success = fix_forest_py()
    if success:
        print("\n修复完成！请尝试重新运行您的DeepForest代码")
    else:
        print("\n修复失败，请考虑降级NumPy版本或联系DeepForest开发人员") 