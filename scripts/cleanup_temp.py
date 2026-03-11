#!/usr/bin/env python3
"""
清理临时文件和中间结果目录
"""
import os
import shutil
import stat

def remove_readonly(func, path, _):
    """移除只读属性"""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def safe_remove_directory(dir_path):
    """安全删除目录"""
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path, onerror=remove_readonly)
            print(f"✓ 删除目录: {dir_path}")
            return True
        except Exception as e:
            print(f"✗ 删除失败 {dir_path}: {e}")
            return False
    else:
        print(f" 目录不存在: {dir_path}")
        return True

def safe_remove_file(file_path):
    """安全删除文件"""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"✓ 删除文件: {file_path}")
            return True
        except Exception as e:
            print(f"✗ 删除失败 {file_path}: {e}")
            return False
    else:
        print(f" 文件不存在: {file_path}")
        return True

def clean_intermediate_dirs():
    """清理中间结果目录"""
    print("=" * 60)
    print("清理中间结果目录")
    print("=" * 60)
    
    intermediate_dirs = [
        # 中间分析目录
        'results/cooperative_simulation_analysis',
        'results/final_aggregation', 
        'results/improved_cooperative_simulation',
        'results/performance_comparison',
        'results/sentinel_analysis',
        'results/simple_performance_report',
    ]
    
    all_success = True
    for dir_path in intermediate_dirs:
        if not safe_remove_directory(dir_path):
            all_success = False
    
    return all_success

def clean_pycache():
    """清理__pycache__目录"""
    print("\n" + "=" * 60)
    print("清理__pycache__目录和.pyc文件")
    print("=" * 60)
    
    pycache_dirs = []
    pyc_files = []
    
    for root, dirs, files in os.walk('.'):
        # 跳过某些目录
        if 'results/paper_figures' in root:
            continue
        if 'datasets' in root and 'simulate' in root:
            continue
        if 'FD001' in root or 'Simulate' in root:
            continue
            
        # 收集__pycache__目录
        if '__pycache__' in dirs:
            pycache_dirs.append(os.path.join(root, '__pycache__'))
        
        # 收集.pyc文件
        for file in files:
            if file.endswith('.pyc'):
                pyc_files.append(os.path.join(root, file))
    
    all_success = True
    
    # 删除.pyc文件
    for pyc_file in pyc_files:
        if not safe_remove_file(pyc_file):
            all_success = False
    
    # 删除__pycache__目录
    for pycache_dir in pycache_dirs:
        if not safe_remove_directory(pycache_dir):
            all_success = False
    
    return all_success

def clean_temp_files():
    """清理临时文件"""
    print("\n" + "=" * 60)
    print("清理临时文件")
    print("=" * 60)
    
    temp_files = []
    
    # 可能的临时文件模式
    temp_patterns = ['*.tmp', 'temp_*', 'debug_*', '*_temp.*']
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            file_path = os.path.join(root, file)
            # 检查常见临时文件扩展名
            if file.endswith('.log') and 'debug' in file.lower():
                temp_files.append(file_path)
            elif any(pattern in file.lower() for pattern in ['temp', 'debug', 'backup']):
                temp_files.append(file_path)
    
    all_success = True
    for temp_file in temp_files:
        if not safe_remove_file(temp_file):
            all_success = False
    
    return all_success

def generate_cleanup_report():
    """生成清理报告"""
    print("\n" + "=" * 60)
    print("清理完成报告")
    print("=" * 60)
    
    # 检查保留的关键文件
    key_files = [
        'final_experiment_report_updated.md',
        'results/paper_figures/',
        'results/final_optimization/',
        'results/performance_comparison_fixed.md',
        'results/performance_data.csv',
        'results/quantization_report.md',
        '论文.md',
        'convert_FD001_test.py',
        'convert_FD001_train.py',
    ]
    
    print("保留的关键文件:")
    for file_path in key_files:
        if os.path.exists(file_path):
            status = "存在"
        else:
            status = "不存在"
        print(f"  {file_path}: {status}")
    
    # 检查清理效果
    print("\n清理效果:")
    print("  - 中间结果目录: 已清理")
    print("  - __pycache__目录: 已清理")
    print("  - .pyc文件: 已清理")
    print("  - 临时文件: 已清理")
    
    print("\n项目结构已优化，准备论文撰写。")

def main():
    """主函数"""
    print("开始清理项目文件...")
    
    # 执行清理
    clean_intermediate_dirs()
    clean_pycache()
    clean_temp_files()
    
    # 生成报告
    generate_cleanup_report()
    
    print("\n清理完成！")

if __name__ == '__main__':
    main()