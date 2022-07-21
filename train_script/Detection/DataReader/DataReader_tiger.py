import copy
import numpy as np
import cv2
import skimage.io
import torch
import os
from scipy.io import loadmat
import skimage.io as skio
from torch.utils.data import DataLoader,Dataset
import albumentations as albu
from tqdm import tqdm
import json

import skimage
import skimage.io as skio
import scipy.io
from PIL import Image
import csv

project_root = os.path.split(__file__)[0]
print(project_root)

class DatasetReader(Dataset):    
    def __init__(self,dataset_path=r'./data_new_image',type = "train",transforms = None):
        self.transf = transforms  
        self.images_result = []  
        self.masks_result = []  
        self.annotation_center = []  
        self.image_name_result = []  
        self.type = type
        self.images_root = dataset_path

        file_path = os.path.join(dataset_path, 'tiger-coco.json')
        with open(file_path) as f:
            json_Data = json.load(f)
        json_images = json_Data['images']
        json_annotations = json_Data['annotations']

        image_id = {}  
        for im_id in json_images:
            # print(json_images) 1873
            id = im_id['id']
            file_name = im_id['file_name'].split('/')[-1]
            temp_dict = {id:file_name}
            image_id.update(temp_dict)


        data_images_path = os.path.join(self.images_root,self.type)
        images_name = os.listdir(data_images_path)
        print("images_name.size",len(images_name))
        for image_name in images_name:
            image_path = os.path.join(data_images_path,image_name)


            encode_ims = skimage.io.imread(image_path)

            new_image_name = image_name
            if image_name[0:7] == "change_":
                new_image_name = image_name[7:]
            elif image_name[0:7] == "changeA":
                new_image_name = image_name[8:]
            im_id = [k for k,v in image_id.items() if v == new_image_name][0]

            annotation_box = []
            annotation_center = []
            annotation_box.extend([annotation['bbox'] for annotation in json_annotations if annotation['image_id'] == im_id])


            for box in annotation_box:
                annotation_center.append([box[0] + box[2] / 2, box[1] + box[3] / 2])   

            mask_det = np.zeros((encode_ims.shape[0], encode_ims.shape[1], 1), dtype=np.uint8)

            for (x, y) in annotation_center:
                x = np.round(x).astype(np.int32)
                y = np.round(y).astype(np.int32)
                cv2.circle(mask_det, center=(x , y), radius=3, color=1, thickness=-1)

            self.images_result.append(encode_ims)
            self.masks_result.append(mask_det)
            self.image_name_result.append(image_name)
            self.annotation_center.append(annotation_center)


    def __getitem__(self, item):  


        image = self.images_result[item]                    
        mask_det = self.masks_result[item]                  
        image_name = self.image_name_result[item]           
        annotation_center = self.annotation_center[item]    

        img_mask_enhance = self.transf(image=image, mask=mask_det)
        image = img_mask_enhance['image']  
        mask_det = img_mask_enhance['mask']  

        image = np.transpose(image / 255.0, (2, 0, 1)).astype(np.float32)     
        mask_det = np.transpose(mask_det, (2, 0, 1)).astype(np.float32)

        return image, mask_det, annotation_center, image_name

    def __len__(self):
        return len(self.images_result)

if __name__ == '__main__':
    transform = albu.Compose([

        albu.RandomCrop(64,64,always_apply=False, p=1),

    ])  


    def collate_fn(batch):
        
        batch = list(zip(*batch))
        image = torch.tensor(batch[0])
        mask_det = torch.tensor(batch[1])
        det_center_dict = batch[2]
        image_name = batch[3]
        del batch
        return image, mask_det, det_center_dict, image_name


    batch_size = 64
    data_path = '/YOUR_DIR/data_new_image/'
    train_dataset = DatasetReader(data_path,type = "train",transforms = transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=collate_fn)

    for epoch in range(100):
        for index,(image, mask_det, annotation_center, image_name) in enumerate(train_loader):
            print(image_name)
