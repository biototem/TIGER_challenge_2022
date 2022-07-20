from typing import List

import torch

from basic import join
from .mod_A import mod_A

# m = mod_A()
# ims = torch.rand([3, 3, 128, 128])
# feats = m.do_feats(ims)
# heats = m.do_heats(feats)
# print(heats.shape)

M = mod_A(path=join('~/env/wt_s_0_checkpoint.pt'))


def wt_interface_trans_img(img: torch.Tensor) -> torch.Tensor:
    assert img.shape == (128, 128, 3)
    return M.do_feats(img.permute(2, 0, 1).unsqueeze(0) / 255)[0, :]


def wt_predict(character_list: List[torch.Tensor]) -> List[float]:
    character_list = torch.stack(character_list, dim=0)
    attention_list = M.do_heats(character_list)
    return attention_list.tolist()
