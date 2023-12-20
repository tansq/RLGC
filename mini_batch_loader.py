import os
import numpy as np
import cv2
import torch
from torchvision import transforms

class MiniBatchLoader(object):
 
    def __init__(self, train_path, test_path, image_dir_path, img_length, img_width):
 
        # load data paths
        self.training_path_infos = self.read_paths(train_path, image_dir_path)
        self.testing_path_infos = self.read_paths(test_path, image_dir_path)
 
        self.img_length = img_length
        self.img_width = img_width
        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    # test ok
    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            src_full_path = os.path.join(src_path, line)
            if os.path.isfile(src_full_path):
                yield src_full_path
 
    # test ok
    @staticmethod
    def count_paths(path):
        c = 0
        for _ in open(path):
            c += 1
        return c
 
    # test ok
    @staticmethod
    def read_paths(txt_path, src_path):
        cs = []
        for pair in MiniBatchLoader.path_label_generator(txt_path, src_path):
            cs.append(pair)
        return cs
 
    def load_training_data(self, indices):
        return self.load_data(self.training_path_infos, indices, augment=True)
 
    def load_testing_data(self, indices):
        return self.load_data(self.testing_path_infos, indices)
 
    
    # test ok
    
    
    
    def load_data(self, path_infos, indices, augment=False):
        mini_batch_size = len(indices)
        in_channels = 3
        masks = np.zeros((mini_batch_size, 1, self.img_length, self.img_width)).astype(np.float32)
        path_list = []
        raw_x = np.zeros((mini_batch_size, in_channels, self.img_length, self.img_width)).astype(np.float32)
        if mini_batch_size>1:    
            for i, index in enumerate(indices):
                path = path_infos[index]
                mask_path=path.replace('.png','_mask.png')
                #img = cv2.imread(path).astype(np.float32)[..., ::-1]
                img = cv2.imread(path).astype(np.float32)
                mask = cv2.imread(mask_path,0).astype(np.float32) / 255.
                raw_x[i,:,:,:]  = img.transpose(2,0,1)
                #cv2.imwrite("./ori/"+path.split('/')[-1],img)
                #cv2.imwrite("./trans/"+path.split('/')[-1], raw_x[i,:,:,:].transpose(1,2,0))
                #img = img.transpose(2,0,1)
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))
                masks[i,0,:,:] = mask
                path_list.append(path)
        else: 
            path = path_infos[indices[0]]
            mask_path = path.replace('.png','_mask.png')
            img = cv2.imread(path).astype(np.float32)[..., ::-1]
            #img = cv2.imread(path).astype(np.float32)
            mask = cv2.imread(mask_path,0).astype(np.float32) / 255.
            
            if img is None:
                raise RuntimeError("invalid image")
            raw_x[0,:,:,:] = img.transpose(2,0,1)
            #raw_x[0,:,:,:] = img
            #img = img.transpose(2,0,1)
            masks[0,0,:,:] = mask
            path_list.append(path)
 
        return  masks, path_list, raw_x
