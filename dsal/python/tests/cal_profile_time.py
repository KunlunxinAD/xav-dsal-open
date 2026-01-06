import re
from typing import Optional, Union

def calculate_xpurt_total_time(
    input_source: Union[str, list],
    input_type: str = "file",  # "file" 读取文件，"text" 直接输入日志文本（列表/字符串）
    output_unit: str = "ms"   # 输出单位："ns"（纳秒）或 "ms"（毫秒）
) -> None:
    """
    统计日志中 [XPURT_PROF] 条目的总耗时
    
    Args:
        input_source: 输入源（文件路径 或 日志文本字符串/列表）
        input_type: 输入类型，"file" 或 "text"
        output_unit: 输出单位，"ns" 或 "ms"
    """
    # 正则表达式：匹配 [XPURT_PROF] 行末尾的纳秒数（最后一列数字）
    xpurt_pattern = re.compile(r'\[XPURT_PROF\].*\s+(\d+)\s*ns$')
    total_ns = 0
    matched_count = 0  # 匹配到的条目数量
    matched_times = []  # 存储所有匹配到的纳秒时间（便于核对）

    # 1. 读取输入内容
    lines = []
    if input_type == "file":
        try:
            with open(input_source, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"错误：文件 {input_source} 不存在！")
            return
        except Exception as e:
            print(f"错误：读取文件失败 - {str(e)}")
            return
    elif input_type == "text":
        if isinstance(input_source, str):
            lines = input_source.split('\n')  # 字符串按换行分割为列表
        elif isinstance(input_source, list):
            lines = input_source  # 直接使用列表
        else:
            print("错误：input_source 为 'text' 类型时，必须是字符串或列表！")
            return
    else:
        print("错误：input_type 只能是 'file' 或 'text'！")
        return

    # 2. 匹配并提取纳秒时间
    for line in lines:
        line = line.strip()  # 去除首尾空白（避免换行符/空格干扰）
        match = xpurt_pattern.search(line)
        if match:
            ns = int(match.group(1))
            total_ns += ns
            matched_count += 1
            matched_times.append(ns)

    # 3. 单位转换
    if output_unit == "ms":
        total = total_ns / 1e6   # 纳秒 → 毫秒（1ms = 1e6 ns）
        unit = "毫秒 (ms)"
    elif output_unit == "ns":
        total = total_ns
        unit = "纳秒 (ns)"
    else:
        print("错误：output_unit 只能是 'ns' 或 'ms'！")
        return

    # 4. 输出结果
    print("=" * 60)
    print(f"[XPURT_PROF] 耗时统计结果")
    print("=" * 60)
    print(f"匹配到的条目数量：{matched_count} 个")
    print(f"所有耗时（纳秒）：{matched_times}")
    print(f"-" * 60)
    print(f"总耗时：{total:.6f} {unit}")
    if output_unit == "ms":
        print(f"（等价于 {total_ns:,} 纳秒）")
    print("=" * 60)


# ------------------------------
# 用法示例（根据你的场景选择一种）
# ------------------------------

if __name__ == "__main__":
    # 用法1：读取日志文件（推荐，将日志保存为文件后使用）
    # 请将下面的路径替换为你的日志文件实际路径
    calculate_xpurt_total_time(
        input_source="1208.log",  # 日志文件路径
        input_type="file",
        output_unit="ms"  # 输出毫秒（可改为 "ns" 输出纳秒）
    )