import csv
import json
import os
from typing import Any
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import cv2

from basic import config, join
from utils import Timer, Drawer, SimpleMultiPolygon
from .evaluate import merge_evaluate, info_evaluate
from .info import create_info
from .merger import TorchMerger
from .divider import divide
from .score_table import csv_evaluate_old, csv_evaluate


class PredictManager(object):
    """
    测试图预测流程,具体的讲,包括:
    1. 大图块预测
        > patch 预测
        > 高斯融合
    2. 实例分割
        > 边界响应
        > otsu二值化
        > 实例标记
        > 分水岭聚类
    3. 轮廓提取
        > 多边形边界捕获
        > 实例类型标记
        > 信息统计
        > 结果输出(XML and 示意图)
    4. 结果评估
        > 轮廓匹配
        > 非极大值抑制
        > 评估统计
        > 结果可视化
    """

    def root(self, root: str):
        self.__root__ = root
        self.drawer = Drawer(root=root)

    def predict(self, model: torch.nn.Module, predict_set):
        self.T.track(' -> start predict ...')
        T = self.T.tab()
        model.eval()
        merged_set = {}
        for subset in predict_set:
            T.track(f' -> make predict {subset.name}')
            loader = DataLoader(
                    subset,
                    batch_size=config['train.batch_size'],
                    num_workers=config['train.num_workers']
            )
            h, w = subset.get_HW()
            with self.merger.with_shape(W=w, H=h):
                self.__predict__(model, loader, T=T)
                result = self.merger.tail()
                merged_set[subset.name] = result
        return merged_set

    def __predict__(self, model, loader, T: Timer = None):
        for i, (grids, patches) in enumerate(loader):
            # if T: T.track(f'\t -> patch {i}: {len(patches)}')
            results = model(patches.cuda())
            self.merger.set(results, grids)

    def instance(self, merged_set, visual=False):
        self.T.track(' -> start instance ...')
        T = self.T.tab()
        instance_set = {}
        for name, predict in merged_set.items():
            T.track(f' -> make instance {name}')
            result = divide(predict, T=T.tab(), show=False)
            instance_set[name] = result
            if visual:
                T.track(f'\t -> visual instance {name}')
                self.drawer.name(f'{name}').instance(instance=result)
        return instance_set

    def contours(self, instance_set):
        """
        cv2.findContours 的 contours 格式: tuple[array] -> array(points, 1, (y, x))
        我的封装为: list[array] -> array(points, (y, x))
        """
        self.T.track(' -> start contours ...')
        T = self.T.tab()
        contours_set = {}
        for name, instance in instance_set.items():
            T.track(f' -> make contours {name}')
            # instance_map = np.zeros(shape=(1000, 1000), dtype=np.uint8)
            # instance_map[500: 600, 100:200] = 1
            instance_map = (instance > 2).astype(np.uint8)
            instance_map = cv2.erode(instance_map, cv2.getStructuringElement(cv2.MORPH_ERODE, ksize=(3, 3)))
            # cv2 格式, Tuple[contour * contour_number] -> contour(point_number, 1, (y, x))
            contours, _ = cv2.findContours(image=instance_map, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            # shapely 格式 List[contour * contour_number] -> contour[point(y, x) * point_number]
            # 这里, 只保留外轮廓
            contours = list(c[:, 0, :] for c in contours if len(c) >= 3)
            # Shapely 对象 (经过包装的) -> 本项目后续全部参数传递均使用该包装对象
            contours = SimpleMultiPolygon(outers=contours)
            # 轮廓开操作 -> 用于除去那些奇怪的点、线
            contours = contours.smooth(distance=3)
            # 轮廓简化 -> 用于减轻交并运算的运算量
            contours = contours.simplify(0.5)
            contours_set[name] = SimpleMultiPolygon.asSimple(contours)
        return contours_set

    def info(self, contours_set, merged_set, log=False):
        self.T.track(' -> start info ...')
        T = self.T.tab()
        info_set = {}
        for name, contours in contours_set.items():
            T.track(f' -> make info {name}')
            result = create_info(contours, merged_set[name], show=False)
            info_set[name] = result
            if log:
                os.makedirs(join(self.__root__, 'info_logs'), exist_ok=True)
                with open(join(self.__root__, 'info_logs', f'{name}.txt'), 'w') as f:
                    json.dump(result, f, indent=4)
        return info_set

    def output(self, info_set):
        # TODO: 这里输出 xml
        pass

    def evaluate_old(self, merged_set, pset, log=True, suffix=''):
        self.T.track(' -> start evaluate with merge_set ...')
        T = self.T.tab()
        evaluate_set = {}
        for name, predict in merged_set.items():
            T.track(f' -> make evaluate {name}{suffix}')
            result = merge_evaluate(predict, pset.get_label(name), box=pset.get_box(name), T=T)
            evaluate_set[name] = result
            # print(json.dumps(result, indent=4))
            if log:
                os.makedirs(join(self.__root__, 'evaluate'), exist_ok=True)
                csv_evaluate_old(join(self.__root__, 'evaluate', f'{name}{suffix}.csv'), result)
        return evaluate_set

    def evaluate(self, info_set, pset, log=True, visual=False):
        self.T.track(' -> start evaluate with info_set ...')
        T = self.T.tab()
        for name, info in info_set.items():
            T.track(f' -> make evaluate {name}')
            result = info_evaluate(info, pset.get_label(name), box=pset.get_box(name), T=T)
            # print(json.dumps(result, indent=4))
            if log:
                os.makedirs(join(self.__root__, 'evaluate'), exist_ok=True)
                csv_evaluate(join(self.__root__, 'evaluate', f'{name}.csv'), result)
            if visual:
                predict = info2merged(info=info)
                self.drawer.name(f'{name}-contour')
                self.drawer.predict_label_iou(predict=predict, label=pset.get_label(name), box=pset.get_box(name))

    def evaluate_all(self, evaluate_set, suffix=''):
        with open(join(self.__root__, 'evaluate', f'sum_to_all{suffix}.csv'), 'w') as fp:
            f = csv.writer(fp, delimiter=';')
            f.writerow(('class', 'img_dice(lab>0)', 'count(pre>100)', 'count(lab>0)', 'count(dice>0.2)', 'pix_dice', 'sum(prd)', 'sum(lab)', 'sum(inter)'))
            prs = np.zeros(shape=6, dtype=np.float32)
            gts = np.zeros(shape=6, dtype=np.float32)
            inters = np.zeros(shape=6, dtype=np.float32)
            dices = np.zeros(shape=6, dtype=np.float32)
            ct_pres = np.zeros(shape=6, dtype=np.float32)
            ct_labs = np.zeros(shape=6, dtype=np.float32)
            ct_dices = np.zeros(shape=6, dtype=np.float32)
            for evaluate in evaluate_set.values():
                pr = evaluate['pre_pixes']
                gt = evaluate['lab_pixes']
                inter = evaluate['inter_pixes']
                dice = evaluate['dice_pixes']
                prs += pr
                gts += gt
                inters += inter
                dices += dice
                ct_pres += [p > 100 for p in pr]
                ct_labs += [g > 0 for g in gt]
                ct_dices += [d > 0.2 for d in dice]

            for i, cls in enumerate(['bg', 'stroma', 'normal', 'tumor', 'necrosis', 'vessel']):
                f.writerow((
                    cls,
                    '%.3f' % (dices[i] / (ct_labs[i] + 1e-19)),
                    '%.3f' % ct_pres[i],
                    '%.3f' % ct_labs[i],
                    '%.3f' % ct_dices[i],
                    '%.3f' % (2 * inters[i] / (prs[i] + gts[i] + 1e-19)),
                    '%.3f' % prs[i],
                    '%.3f' % gts[i],
                    '%.3f' % inters[i],
                ))
        return {
            'pr': prs.tolist(),
            'gt': gts.tolist(),
            'inter': inters.tolist(),
            'dice': dices.tolist(),
            'ct_pre': ct_pres.tolist(),
            'ct_lab': ct_labs.tolist(),
            'ct_dice': ct_dices.tolist(),
        }

    def visual_old(self, merged_set, pset, suffix=''):
        T = self.T.tab()
        for name, predict in merged_set.items():
            T.track(f' -> visualizing {name}{suffix}')
            h, w = pset.get_HW(name)
            image = pset.get_image(name)
            label = pset.get_label(name)
            box = pset.get_box(name)
            # 图片太大,进行 dot 运算可能导致卡死, 所以要缩放
            if h > 5000 or w > 5000:
                rs = max(h, w) / 5000
                nh = round(h / rs)
                nw = round(w / rs)
                T.track(f' -> resizing from {(w, h)} to {(nw, nh)} for visual')
                predict = cv2.resize(predict, (nw, nh))
                image = cv2.resize(image, (nw, nh))
                label = cv2.resize(label.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
                box = list(map(lambda q: round(q/rs), box))
            self.drawer.name(f'{name}{suffix}')
            T.track(f' -> visual predict {name}{suffix}')
            self.drawer.predict(predict=predict, box=box)
            T.track(f' -> visual image-predict {name}{suffix}')
            self.drawer.image_predict(image=image, predict=predict, box=box)
            T.track(f' -> visual image-label {name}{suffix}')
            self.drawer.image_label(image=image, label=label, box=box)
            T.track(f' -> visual predict-label {name}{suffix}')
            self.drawer.predict_label(predict=predict, label=label, box=box)
            T.track(f' -> visual image-predict-label {name}{suffix}')
            self.drawer.image_predict_label(image=image, predict=predict, label=label, box=box)

    def visual(self, info_set, pset):
        self.T.track(' -> start visual ...')
        T = self.T.tab()
        for name, info in info_set.items():
            T.track(f' -> make visual {name}')
            # TODO: info 绘图应当在轮廓上标记一些信息，例如面积大小之类的
            self.drawer.name(name).info(info)
            self.drawer.name(name).image_info(info=info, image=pset.get_image(name))

    def __init__(self, channel: int = 0, ksize: int = 0, T: Timer = None):
        """
        __init__, __enter__, __exit__ 用于实现 with 语法,提供状态管理
        """
        self.T = T or Timer()
        self.G = torch.no_grad()
        self.merger = TorchMerger(channel=channel, ksize=ksize, kernel_steep=6)
        self.__root__ = join('~/output/predict')
        self.drawer = Drawer(root=self.__root__)

    def __enter__(self):
        self.G.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        self.G.__exit__(exc_type, exc_value, traceback)
        return False


def info2merged(info: dict):
    h, w = info['size']
    normal = np.zeros(shape=(h, w), dtype=np.uint8)
    atrophic = np.zeros(shape=(h, w), dtype=np.uint8)
    contours = info['contours']
    for contour in contours:
        coords = contour['contour']
        tp = normal if contour['base']['tp'] == 'normal' else atrophic
        cv2.fillPoly(tp, [np.array(coords, dtype=np.int32)], 1, lineType=None)
    result = np.stack([np.zeros_like(normal), np.zeros_like(normal), normal, atrophic], axis=2)
    return result.astype(np.float32)
