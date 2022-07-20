"""
valid-metrics 是 metrics 的转写。
在新的验证流程中，对模型性能的评估基于高斯融合后的大图块进行，这产生了两种并行的评估系统：
    1. 基于大图块的评估
        大图块的格式为 numpy [h, w, c] -> float32 [0, 1]
        这与 epoch 中的 torch 格式不兼容
    2. 基于像素的评估
        在大图块的基础上，基于像素的评估要求存储全部大图块的像素级信息，这是不现实的
        因此，必须将 metrics 方法完全拆解，根据底层逻辑重建一套具有状态缓存的方法
这就是这篇代码的目的
"""
from basic import InterfaceProxy, hyp
from utils import Assert


# 请放心，任何表达式均会得到妥善计算，语法如下
#   [pr, gt, tp, fp, tn, fn, ce, dice_loss, mf, dice]
#   其中 mf 必须包含括号参数
# 其中， pr、gt、tp 是原始值数组，fp、tn、fn 是计算值数组，ce、dice_loss、mf、dice 是计算值单值


class ValidMetric(object):
    def __init__(self, **kwargs):
        """
        metric 表达式语法:
            1. 字符串必须是一个表达式
            2. 可用关键词：
                pixes: 统计像素数 数组值
                pr: 预测像素数 数组值
                gt: 标签像素数 数组值
                tp/inter： 正确正样本预测像素数 数组值
                fp: 错误正样本预测像素数 数组值（非原始值，不建议使用）
                tn: 正确负样本预测像素数 数组值（非原始值，不建议使用）
                fp: 错误负样本预测像素数 数组值（非原始值，不建议使用）
                lce: 对类计算 交叉熵损失 得到 标量值， 或 lce{} 得到数组值， 或 lce[] 在指定类上计算
                ldice: 对类计算 dice损失，同上
                mf: 对类计算 mf-score评估，同上
                dice: 对类计算 dice评估，同上
                .sum: 对类求代数和，必须由数组值发起，可指定参与运算的 class，例： pr.sum[1,2] + gt.sum
                .mean: 对类求代数平均，同上
                .gmean: 对类求几何平均，同上
            3. 解析规则：括号
                表达式支持小括号（）、方括号[]和大括号{}，其中:
                    小括号仅用于改变运算优先级，不可省略乘法符号
                    方括号仅用于 数组值 转 标量值 时的的类别选择，得到标量值
                    大括号仅用于 数组值 的类别选择，得到数组值
                    注意：有且只有 方括号[] 能将数组值变为标量值
            4. 原理与实现：
                字符串表达式将被解析为一个 lambda 函数，伴随其它参数一并存入状态池中
                当评估一张大图块时，表达式涉及的原始值将被存入状态池中，这一过程伴随着 pixes 的过滤
                注意： pixes-filter 影响 image 评估，但 image-filter 不影响 pixes 评估！

        """
        # 解析参数
        p = {
            # 应当由参数传递
            'uses': {
                'any_name_in_logs(such like:"loss")': {
                    'metric': '2 * ce + 7 * mf[0.5]',
                    'image-reduce': 'sum',   # 大图块间合并方法， 支持 sum 代数和, mean 代数均值, gmean 几何均值
                    'image-filter': 'gt[2] > 100 and pr.sum[2,3,4] < 1e5',  # 滤除不满足条件的大图块
                    'pixes-filter': 'gt.sum > 0',   # 滤除不满足条件的像素
                    'ce-limit': 15,  # 计算交叉熵时的 limit 限值
                    'ce-smooth': 0,  # 计算交叉熵时的 label 平滑值
                    'ce-weight': (1, ) * hyp['dataset.class_num'],  # 计算交叉熵最终值时的权重
                    'mf-beta': 2,   # 计算 mf-score 时的 beta 值
                }
            }
        }
        p.update(**kwargs)
        Assert.not_none(**p)

