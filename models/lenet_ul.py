import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet_UL(nn.Module):
    def __init__(self, num_classes,in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=6,
                               kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)

        self.fc_1 = nn.Linear(16 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, 84)
        # self.fc_1_ul = nn.Linear(16 * 4 * 4, 120)
        # self.fc_2_ul = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, num_classes)
        self.classifier_ul = nn.Linear( 84, num_classes)

    def forward(self, x):

        # x = [batch size, 1, 28, 28]

        x = self.conv1(x)

        # x = [batch size, 6, 24, 24]

        x = F.max_pool2d(x, kernel_size=2)

        # x = [batch size, 6, 12, 12]

        x = F.relu(x)

        x = self.conv2(x)

        # x = [batch size, 16, 8, 8]

        x = F.max_pool2d(x, kernel_size=2)

        # x = [batch size, 16, 4, 4]

        x = F.relu(x)

        x = x.view(x.shape[0], -1)

        # x = [batch size, 16*4*4 = 256]

        h = x

        x_norm = self.fc_1(x)
        # x_ul=self.fc_1_ul(x)

        # x = [batch size, 120]

        x_norm = F.relu(x_norm)
        # x_ul=F.relu(x_ul)

        x_norm = self.fc_2(x_norm)
        # x_ul = self.fc_2_ul(x_ul)
        # x = batch size, 84]

        x_norm = F.relu(x_norm)
        # x_ul=F.relu(x_ul)

        # x = self.classifier(x)

        a = self.classifier(x_norm)
        b = self.classifier_ul(x_norm)
        z = torch.cat((a,b),dim=1)

        # x = [batch size, output dim]

        return z
    
def lenet_ul(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = LeNet_UL(**kwargs)
    return model