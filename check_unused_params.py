#!/usr/bin/env python3
"""
检查域适应训练脚本中未使用的参数
"""

import re
import os


def check_unused_parameters():
    """检查未使用的参数"""
    script_path = "/home/ytao/Thesis/src/bvm_training/trans_bvm_self_supervised_thesis/train.py"

    print("🔍 检查域适应训练脚本中的参数使用情况")
    print("=" * 60)

    # 读取文件内容
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 提取所有定义的参数
    param_pattern = r'parser\.add_argument\(\s*"--(\w+)"'
    defined_params = re.findall(param_pattern, content)

    print(f"📋 定义的参数总数: {len(defined_params)}")
    print()

    # 检查每个参数的使用情况
    unused_params = []
    used_params = []

    for param in defined_params:
        # 检查参数是否在代码中被使用（除了定义处）
        usage_pattern = rf"opt\.{param}\b"
        usage_matches = re.findall(usage_pattern, content)

        # 过滤掉在打印配置中的使用（这些不算实际功能使用）
        functional_usage = []
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if re.search(usage_pattern, line):
                # 检查是否是在打印语句中
                if not any(x in line for x in ['print(f"', "print(", "help="]):
                    functional_usage.append((i + 1, line.strip()))

        if functional_usage:
            used_params.append({"name": param, "usage_count": len(functional_usage), "usage_lines": functional_usage[:3]})  # 只显示前3个使用位置
        else:
            unused_params.append(param)

    # 显示结果
    print("✅ 已使用的参数:")
    for param in used_params:
        print(f"  {param['name']}: {param['usage_count']} 次使用")
        if param["usage_count"] <= 3:
            for line_num, line in param["usage_lines"]:
                print(f"    L{line_num}: {line[:80]}...")
        print()

    if unused_params:
        print("⚠️  未使用的参数:")
        for param in unused_params:
            print(f"  - {param}")
        print()
        print("💡 建议: 可以考虑删除这些未使用的参数以保持代码整洁")
    else:
        print("🎉 所有参数都有被使用！")

    print("\n📊 统计:")
    print(f"  总参数数: {len(defined_params)}")
    print(f"  已使用: {len(used_params)}")
    print(f"  未使用: {len(unused_params)}")

    return unused_params


if __name__ == "__main__":
    unused = check_unused_parameters()

    if unused:
        print(f"\n🔧 发现 {len(unused)} 个未使用的参数，建议进行清理")
    else:
        print(f"\n✅ 参数使用检查完成，代码整洁！")
