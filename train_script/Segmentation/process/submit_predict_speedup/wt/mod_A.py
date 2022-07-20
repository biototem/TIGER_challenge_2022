import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torchvision.models.resnet import resnet50
from torchvision.transforms import Normalize
try:
    from model_clam import CLAM_SB
except:
    from .model_clam import CLAM_SB
from scipy.stats import rankdata


class mod_A(nn.Module):
    def __init__(self, path: str = None):
        super().__init__()

        self.feat_extract = resnet50(pretrained=True)
        self.feat_extract.layer4 = nn.Identity()
        self.feat_extract.fc = nn.Identity()

        self.clam = CLAM_SB(dropout=True, size_arg='small', n_classes=2)
        path = path or os.path.dirname(__file__) + '/s_0_checkpoint.pt'
        clam_ck = torch.load(path, 'cpu')
        del clam_ck['instance_loss_fn.labels']
        self.clam.load_state_dict(clam_ck)

        self.feat_extract.eval()
        self.clam.eval()

    @staticmethod
    def to_percentiles(scores):
        scores = rankdata(scores, 'average').astype(np.float32) / len(scores)
        return scores

    @torch.inference_mode()
    def do_feats(self, ims):
        '''
        输入要求为 torch.Tensor，通道顺序为 BCHW，值范围为 [0, 1]
        输出为 numpy 数组
        :param ims:
        :return:
        '''
        # ims shape BCHW, [0, 1]
        assert isinstance(ims, torch.Tensor)
        assert ims.ndim == 4 and ims.shape[1] == 3

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        ims = Normalize(mean, std)(ims)

        feats = self.feat_extract(ims)

        assert feats.ndim == 2 and feats.shape[1] == 1024
        return feats

    @torch.inference_mode()
    def do_heats(self, feats):
        '''
        输入要求为 torch.Tensor，通道顺序为BC，C通道维度为1024
        输出为 numpy 数组，是归一化后分数，值域为 [0, 1]
        :param feats:
        :return:
        '''
        assert isinstance(feats, torch.Tensor)
        assert feats.ndim == 2 and feats.shape[1] == 1024

        logits, Y_prob, Y_hat, A_raw, results_dict = self.clam(feats)

        Y_hat = Y_hat.item()

        # A = A_raw[Y_hat]
        A = A_raw

        A = A.view(-1, 1).cpu().numpy()
        scores = self.to_percentiles(A)

        return scores


if __name__ == '__main__':
    m = mod_A()
    ims = torch.rand([3, 3, 128, 128])

    feats = m.do_feats(ims)
    heats = m.do_heats(feats)

    print(heats.shape)
