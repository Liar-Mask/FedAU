import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.conv2d import ConvBlock
# from models.layers.passportconv2d_private import PassportPrivateBlock


class BasicPrivateBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):#(512, 512, 2) (512, 512, 1)
        super(BasicPrivateBlock, self).__init__()

        self.convbnrelu_1 = ConvBlock(in_planes, planes, 3, stride, 1)
        self.convbn_2 = ConvBlock(planes, planes, 3, 1, 1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = ConvBlock(in_planes, self.expansion * planes, 1, stride, 0) # input, output, kernel_size=1

    def forward(self, x):
        
        out = self.convbnrelu_1(x)
        out = self.convbn_2(out)

        if not isinstance(self.shortcut, nn.Sequential):
            
            out = out + self.shortcut(x)

        else: # if self.shortcut == nn.Sequential 
            out = out + x
        out = F.relu(out)
        return out

class ResNetPrivate(nn.Module):
    def __init__(self, block, num_blocks, num_classes,in_channels): #BasicPrivateBlock, [2, 2, 2, 2], **model_kwargs
        super(ResNetPrivate, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks
        self.in_channels=in_channels

        self.convbnrelu_1 = ConvBlock(self.in_channels, 64, 3, 1, 1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)
        # self.classifier_ul = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride): #BasicPrivateBlock, planes = 512, numblocks = 2, stride =2, **model_kwargs
        strides = [stride] + [1] * (num_blocks - 1) # [2] + [1]*1 = [2, 1]
        layers = []
        for i, stride in enumerate(strides): #stride = 2 & 1
            layers.append(block(self.in_planes, planes, stride)) # (512, 512, 2)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
       
        out = self.convbnrelu_1(x)

        for block in self.layer1:
            out = block(out)
        for block in self.layer2:
            out = block(out)
        for block in self.layer3:
            out = block(out)
        for block in self.layer4:
            out = block(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

def resnet(**model_kwargs):
    return ResNetPrivate(BasicPrivateBlock, [2, 2, 2, 2], **model_kwargs)


