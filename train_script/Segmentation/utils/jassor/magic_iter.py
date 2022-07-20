import math
from typing import Tuple


def magic_iter(target: int, unit: int, step: int, margin: Tuple[int, int] = (0, 0)):
    """
    魔法迭代的目的是用一组长为 unit 的窗口，以步长 step， 迭代遍历长度为 target 的区间
    在保证 target 被完全覆盖的基础上，可以指定左右的松弛额度 margin， 使窗口彼此不重叠
    :return: iterator(i, x)
            -> i from 1 to n stand for index
            -> x from margin-left to margin-right stand for start-position
    """
    # 当图片太大、窗口放不下时，直接下一位
    if target < unit:
        return

    # x for width
    remainder = (unit - target) % step
    step_count = math.ceil((target - unit) / step)
    if remainder == 0:
        offset = 0
    else:
        left, right = margin
        if left + right < remainder:
            # 宽容度不足，减少步长以近似平铺
            step -= (remainder - left - right) / step_count
        else:
            # 宽容度充足，尝试整倍数平铺
            # 计算左右边界的宽容数
            ratio = min(1 + left, remainder) / min(1 + right, remainder)
            left = int(ratio / (1 + ratio) * remainder)
            right = int(1 / (1 + ratio) * remainder)
            # 消除浮点数取整引起的 0/1 误差
            step = (target - unit + left + right) / step_count
        offset = left
    # 步长数 + 原窗口0计数 => 总迭代量
    for i in range(step_count + 1):
        yield i, round(step * i - offset)
