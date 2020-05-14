########################################
### 2020/05/10 Lawrence
### model for captcha identification
########################################


########################################
###  The model structure should be consist with the size the the image.
###  The size of the image is [Channel:1,Height:40,Width:100].
###  Here is how to calculate the size of tensor after CNN:
###  https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
########################################







import torch.nn as nn

class captcha_identifier(nn.Module):
    def __init__(self, num_class=36, num_char=5):
        super(captcha_identifier, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(
                #batch*1*40*100
                nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                #batch*16*19*49
                nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size = 3),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                #batch*64*8*23
                nn.Conv2d(in_channels = 64, out_channels = 512, kernel_size = 3),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                #batch*512*3*10
                nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                #batch*512*1*8
                )
        self.fc = nn.Linear(512*1*8, self.num_class*self.num_char)
        
    def forward(self, x):
        isDict = isinstance(x,dict)
        if isDict:
            x = x['img']
        
        
        x = self.conv(x)
        x = x.view(-1, 512*1*8)
        x = self.fc(x)
        return x