########################################
### 2020/05/10 Lawrence
### Prediction for captcha
########################################

import torch
import torch.nn as nn
from model import captcha_identifier
from dataloader_captcha import Captcha_Dataset
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plot
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

from mean_std_obtainer import get_mean_std
## this .pth file is the weight of trained model
model_path = './checkpoints/identifier.pth'

source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97+26)]
alphabet = ''.join(source)


## predict function can identify the content of input image.

identifier = captcha_identifier()
identifier.load_state_dict(torch.load(model_path))

transform = transforms.Compose([transforms.ToTensor()])
    ## path = "data/"
dataset = Captcha_Dataset('captcha', transform = transform)

#img = dataset[30]
img = Image.open('pin.png')
  ## the forward function need to 4-dim tensor
#img['img'] = img['img'].unsqueeze(1)
mean,std = get_mean_std('captcha')

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = mean,std = std)])
img = transform(img)
image = img
img = img.unsqueeze(1)
captcha = identifier(img)
captcha = captcha.view(-1,36)

captcha = nn.functional.softmax(captcha, dim=1)
captcha = torch.argmax(captcha, dim=1)
captcha = captcha.view(-1,5)
captcha = captcha.squeeze()

print(captcha)
prediction = ''.join([alphabet[int(i)] for i in captcha.detach().numpy()])
#prediction = predict(dataset[12])
print(prediction)      

plt.imshow(transforms.ToPILImage()(image))
plt.show()
