import torch
import torch.nn as nn
from models.layers.conv2d import ConvBlock
# from models.layers.passportconv2d_private import PassportPrivateBlock

class Unlearn_module(nn.Module):

    def __init__(self, num_classes,in_channels): #in_channels, 
        super().__init__()
        self.num_classes=num_classes
        maxpoolidx = [1, 3, 7]
        layers = []
        inp = in_channels #in_channels
        # oups = {
        #     0: 64,
        #     2: 192,
        #     4: 384,
        #     5: 256,
        #     6: 256
        # }
        # kp = {
        #     0: (5, 2),
        #     2: (5, 2),
        #     4: (3, 1),
        #     5: (3, 1),
        #     6: (3, 1)
        # }
        # for layeridx in range(8):
        #     if layeridx in maxpoolidx:
        #         layers.append(nn.MaxPool2d(2, 2))
        #     else:
        #         k = kp[layeridx][0]
        #         p = kp[layeridx][1]
        #         # if passport_kwargs[str(layeridx)]['flag']:
        #         #     layers.append(PassportPrivateBlock(inp, oups[layeridx], k, 1, p))
        #         # else:
        #         layers.append(ConvBlock(inp, oups[layeridx], k, 1, p))
        #         inp = oups[layeridx]

        # self.features = nn.Sequential(*layers)
        self.classifier_ul = nn.Linear(4*4*256, num_classes)

    def forward(self, x):
        # for m in self.features:
        #     x = m(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_ul(x)

        # a=torch.nn.functional.softmax(x[:,0:int(self.num_classes/2)]) #softmax
        # #print(a.size())
        # b=torch.nn.functional.softmax(x[:,int(self.num_classes/2):self.num_classes])
        #print(b.size())
        # z=torch.cat((a,b),dim=1)
        return x

def unlearn_module(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = Unlearn_module(**kwargs)
    return model
