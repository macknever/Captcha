
########################################
### 2020/05/08 Lawrence
### Dataloader for captcha
########################################


########################################
###  dataloader is similar as the one in project of CPSC 533R
###  
########################################





import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle
import random

source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97+26)]
alphabet = ''.join(source)

class Captcha_Dataset(Dataset):
    def __init__(self, data_folder, transform = None):
        super().__init__()
        self.transform = transform
        if os.path.exists("captcha.npy"):
            self.data = np.load("captcha.npy", allow_pickle=True)
        else:
            self.data = Captcha_Dataset._load_dataset(data_folder)
            self.data.sort(key=lambda x: x["path"])
      
            random.shuffle(self.data)
            np.save("captcha.npy", self.data)

    def __getitem__(self, idx):
        elt = self.data[idx]

        img = elt["img"]
        target = elt["target"]
        
        if self.transform:
            img = self.transform(img)
        target = torch.Tensor(target)
       
        return DeviceDict({"img": img, 
                "target":target}) 

    def __len__(self):
        return len(self.data)

    # @staticmethod
    # def _remove_grayscale(img_list, mask_list):
    #     indices = []
    #     for i, img in enumerate(img_list):
    #         num_channels = img["img"].mode
    #         if num_channels != "RGB":
    #             indices.append(i)
    #     img_list = np.delete(img_list, indices)
    #     mask_list = np.delete(mask_list, indices)
    #     return img_list, mask_list

    @staticmethod 
    def _load_dataset(path):
        captchas = []
        
        for files in os.listdir(path):

            captcha = {}
            img_path = os.path.join(path, files)        
            img = Image.open(img_path)
            #img = img.convert('L')
                ## if the size in the folder is not uniform we need to resize them to make them the same size.
                ##img = img.resize((400, 400))
                
            captcha["img"] = img 

                ## split the name of the file.
                ## the name format is ****.png.
                ## the part we need is ****, it is the content of captcha.
            target_str = files.split('.')[0]
            if len(target_str) != 5:
                print(target_str)
            target = []
            for char in target_str:

                vec = [0]*36 ## 36 means there are 36 characters in the alphabet,0~9,a~z;
                vec[alphabet.find(char)] = 1
                target +=vec
            captcha["target"] = target
            captcha["path"] = img_path
            captchas.append(captcha)
                
        return captchas

    
        

# Wrapper that copies tensors in dict to a GPU
class DeviceDict(dict):
    def __init__(self, *args):
        super(DeviceDict, self).__init__(*args)

    def to(self, device):
        dd = DeviceDict()
        for k, v in self.items():
            if torch.is_tensor(v):
                dd[k] = v.to(device)
            else:
                dd[k] = v
        return dd

transform = transforms.Compose([transforms.ToTensor()])

captcha_path = 'captcha'

captcha_data = Captcha_Dataset(captcha_path, transform=transform)
# print(captcha_data.__len__())
# print(type(captcha_data[0]))
# plt.imshow(transforms.ToPILImage()(captcha_data[40]['img']))
# plt.show()
# print(captcha_data[40]['target'].size())

#plt.imshow(transforms.ToPILImage()(whale_data[268]['mask']))
#plt.show()
