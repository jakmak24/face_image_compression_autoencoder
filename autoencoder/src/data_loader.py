import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageFolder720p(Dataset):

    def __init__(self, folder_path, weight_path = None):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        if weight_path is not None:
            self.weight_files = sorted(glob.glob('%s/*.*' % weight_path))
        else:
            self.weight_files = None
            

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        
        img = np.array(Image.open(path))
        
        h, w, c = img.shape


        pad = ((3, 3), (3, 3), (0, 0))

        # img = np.pad(img, pad, 'constant', constant_values=0) / 255
        img = np.pad(img, pad, mode='edge') / 255.0

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        patches = np.reshape(img, (3,256,256))
        
        
        if self.weight_files is not None:
            weight_file = self.weight_files[index % len(self.weight_files)]
            weight_img = np.array(Image.open( weight_file).convert("L"))
            
            pad = ((3, 3), (3, 3))
                
            weight_img = np.pad(weight_img, pad, mode='edge') / 255.0
            
            weight_img = torch.from_numpy(weight_img).float()  
            weight_patches = np.reshape(weight_img, (256,256))
        else:
            weight_patches = -1
            
        
        
        return img, patches, path, weight_patches


    def get_random(self):
        i = np.random.randint(0, len(self.files))
        return self[i]


    def __len__(self):
        return len(self.files)
