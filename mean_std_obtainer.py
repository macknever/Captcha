########################################
### 2020/05/12 Lawrence
### obtain the mean and std of images in the path
########################################

import os
from PIL import Image
import torch
from torchvision import transforms


def get_mean_std(path):

	means = []
	stds = []

	for files in os.listdir(path):
		img_path = os.path.join(path,files)
		img = Image.open(img_path)

		img = transforms.ToTensor()(img)

		means.append(torch.mean(img))
		stds.append(torch.std(img))

	mean = torch.mean(torch.tensor(means))
	std = torch.mean(torch.tensor(stds))

	return mean,std


