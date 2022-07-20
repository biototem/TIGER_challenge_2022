import numpy as np
# import PIL.Image as Image
from nuclei_det.model.model_tiger import MainNet
from nuclei_det.det_processing import heatmap_nms
from nuclei_det import eval_utils_new
import torch
import torch.utils.data as data
from nuclei_det.merge_util import merger_method
# import torchvision.transforms as transforms
# nuclei_det_model_input_size = 64

if torch.cuda.is_available():
    device = 'cuda:0' 
    # print('Using ' + torch.cuda.get_device_name(0) + '\n')
else:
    device = 'cpu'
nuclei_det_model = MainNet().to(device) # 定义模型
model_name = "./nuclei_det/model_pt/1_model_best_p3.pt"
nuclei_det_model.load_state_dict(torch.load(model_name, 'cpu'))
nuclei_det_model.enabled_b2_branch = True
nuclei_det_model = nuclei_det_model.eval()



def slide_sample_list(image_tile,mask,nuclei_det_model_input_size = 64,step = 0.5):
    assert image_tile.shape[0] == mask.shape[0] and image_tile.shape[1] == mask.shape[1]
    width = image_tile.shape[1]
    height = image_tile.shape[0]
    data_lib = []
    for h in range(0, height, round(nuclei_det_model_input_size * step)):
        for w in range(0, width, round(nuclei_det_model_input_size * step)):
            if np.sum(mask[max(0,h - 8 ):min(h+nuclei_det_model_input_size + 8,image_tile.shape[0]),
                      max(0,w - 8):min(w+nuclei_det_model_input_size + 8,image_tile.shape[1])]) > 0 and \
                w+nuclei_det_model_input_size<=image_tile.shape[1] and \
                h+nuclei_det_model_input_size<=image_tile.shape[0]:
                data_lib.append((w, h))

    return data_lib

class tile_dataloader(data.Dataset):
    def __init__(self, image_tile,mask, nuclei_det_model_input_size = 64,step=0.5,data_lib=None):
        if data_lib is None:
            data_lib = slide_sample_list(image_tile,mask,nuclei_det_model_input_size,step)
        self.image_tile = image_tile
        self.data_lib = data_lib
        self.nuclei_det_model_input_size = nuclei_det_model_input_size
        self.step = step

    def __getitem__(self,index):
        (w,h) = self.data_lib[index]
        exception_flag = 0
        try:
            img = self.image_tile[h:h + self.nuclei_det_model_input_size,
                  w:w + self.nuclei_det_model_input_size,:]
        except Exception:
            img = np.zeros((self.nuclei_det_model_input_size, self.nuclei_det_model_input_size, 3)).astype(np.uint8)
            exception_flag = 1
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255
        return img, exception_flag, w, h

    def __len__(self):
        return len(self.data_lib)

def get_nuclei_det_loader(image_tile,mask,nuclei_det_model_input_size = 64,step=0.5,batch_size = 16):
    dataset = tile_dataloader(image_tile,mask,nuclei_det_model_input_size,step)
    predict_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=False)
    return predict_loader


def loader_prediction(image_tile,mask,nuclei_det_model_input_size = 64,step=0.5,prob_thresh = 0.15,batch_size = 16):
    predict_loader = get_nuclei_det_loader(image_tile,mask,nuclei_det_model_input_size,step,batch_size)
    whole_pred_det_pm_mask = np.zeros((image_tile.shape[0],image_tile.shape[1],2), dtype=np.float32)
    # whole_pred_det_pm_mask[:, :, 1] = 1
    whole_pred_det_2_pm_mask = np.zeros((image_tile.shape[0], image_tile.shape[1], 2), dtype=np.float32)
    # whole_pred_det_2_pm_mask[:, :, 1] = 1
    # tmp_result_list = []
    with torch.no_grad():
        for img,exception_flag,x,y in predict_loader:
            x = x.numpy()
            y = y.numpy()
            exception_flag = exception_flag.numpy()
            img = img.to(device)
            batch_eval_pred_det_temp_list, batch_eval_pred_det_2_temp_list = nuclei_det_model(img)
            valid_batch_pred_det = batch_eval_pred_det_temp_list.softmax(1)[:, 1:]
            valid_batch_pred_det2 = batch_eval_pred_det_2_temp_list.softmax(1)[:, 1:]
            valid_batch_pred = torch.cat([valid_batch_pred_det, valid_batch_pred_det2], 1)
            valid_batch_pred = valid_batch_pred.permute(0, 2, 3, 1).cpu().detach().numpy()
            pred_det_pm, pred_det_2_pm = np.split(valid_batch_pred, [1], -1)
            # 在初检图和终检图相乘之前先调用融合方法
            # pred_det_final_pm = pred_det_pm * pred_det_2_pm
            # pred_det2_pm_final_nms = np.copy(pred_det_final_pm)
            for i in range(len(exception_flag)):
                if exception_flag[i] != 1:
                    whole_pred_det_pm_mask, whole_pred_det_2_pm_mask = merger_method(pred_det_pm[i,:,:,:],pred_det_2_pm[i,:,:,:],
                                                                    whole_pred_det_pm_mask,whole_pred_det_2_pm_mask,
                                                                    x[i],y[i],method = "Average")
            del pred_det_pm,pred_det_2_pm,valid_batch_pred,valid_batch_pred_det,valid_batch_pred_det2,batch_eval_pred_det_temp_list, batch_eval_pred_det_2_temp_list
            torch.cuda.empty_cache()

        whole_pred_det_pm_mask[whole_pred_det_pm_mask[:,:,1]>0,0] = np.float32(whole_pred_det_pm_mask[whole_pred_det_pm_mask[:,:,1]>0,0]/whole_pred_det_pm_mask[whole_pred_det_pm_mask[:,:,1]>0,1])
        whole_pred_det_2_pm_mask[whole_pred_det_2_pm_mask[:, :, 1]>0, 0] = np.float32(whole_pred_det_2_pm_mask[whole_pred_det_2_pm_mask[:, :, 1]>0, 0] / whole_pred_det_2_pm_mask[whole_pred_det_2_pm_mask[:, :, 1]>0, 1])
        whole_pred_det_final_pm = whole_pred_det_pm_mask[:,:,0] * whole_pred_det_2_pm_mask[:,:,0]
        whole_pred_det2_pm_final_nms = np.copy(whole_pred_det_final_pm)
        whole_pred_det2_pm_final_nms = whole_pred_det2_pm_final_nms * heatmap_nms(whole_pred_det2_pm_final_nms)
        whole_pred_det2_pm_final_nms = np.expand_dims(whole_pred_det2_pm_final_nms, axis=2)
        pred_det_post_pts, probs = eval_utils_new.get_pts_from_hm_with_probs(whole_pred_det2_pm_final_nms, prob=prob_thresh)
        pts = np.asarray(pred_det_post_pts, np.float32).reshape([-1, 2])[:, ::-1]
        probs = np.asarray(probs, np.float32).reshape([-1, 1])
        r = np.concatenate([pts, probs], 1)
    
    return r