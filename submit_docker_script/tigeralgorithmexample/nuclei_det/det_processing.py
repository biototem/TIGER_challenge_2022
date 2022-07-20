from nuclei_det.model.model_tiger import MainNet
import numpy as np
import torch
# import torch.nn.functional as F
import cv2

if torch.cuda.is_available():
    device = 'cuda:0' 
    # print('Using ' + torch.cuda.get_device_name(0) + '\n')
else:
    device = 'cpu'
net = MainNet().to(device) # 定义模型
model_name = "./nuclei_det/model_pt/1_model_best_p3.pt"
net.load_state_dict(torch.load(model_name, 'cpu'))
net.enabled_b2_branch = True
net = net.eval()


def heatmap_nms(hm: np.ndarray):
    a = (hm * 255).astype(np.int32)
    a1 = cv2.blur(hm, (3, 3)).astype(np.int32)
    a2 = cv2.blur(hm, (5, 5)).astype(np.int32)
    a3 = cv2.blur(hm, (7, 7)).astype(np.int32)

    ohb = (hm > 0.).astype(np.float32)

    h = a + a1 + a2 + a3

    h = (h / 4).astype(np.float32)

    ht = torch.tensor(h)[None, None, ...]
    htm = torch.nn.functional.max_pool2d(ht, 9, stride=1, padding=4)
    hmax = htm[0, 0, ...].numpy()

    h = (h >= hmax).astype(np.float32)

    h = h * ohb

    h = cv2.dilate(h, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    return h

def process_image_prediction_2(batch_im):
    batch_im = torch.tensor(np.asarray(batch_im),dtype=torch.float32,device=device).permute(0, 3, 1, 2)/255
    with torch.no_grad():

        valid_batch_pred_det, valid_batch_pred_det2 = net(batch_im)
        valid_batch_pred_det = valid_batch_pred_det.softmax(1)[:, 1:]
        valid_batch_pred_det2 = valid_batch_pred_det2.softmax(1)[:, 1:]
        valid_batch_pred = torch.cat([valid_batch_pred_det, valid_batch_pred_det2], 1)
        valid_batch_pred = valid_batch_pred.permute(0, 2, 3, 1).cpu().detach().numpy()
        torch.cuda.empty_cache()
    return valid_batch_pred

########################################################################################################################



