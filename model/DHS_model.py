
import torch
import torch.nn as nn
import torchvision


class RCL(nn.Module):
    """RCL block in DHS net for salient object detection"""

    def __init__(self, channels):
        """Init module.

        :param channels: number of input channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(65, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, smr):
        out1 = self.conv1(x)
        out1 = self.sigmoid(out1)
        out2 = self.sigmoid(smr)
        out = torch.cat((out1, out2), 1)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn(out)  # out_share
        out_share = out
        for i in range(3):
            out = self.conv3(out)
            out = torch.add(out, out_share)
            out = self.relu(out)
            out = self.bn(out)
        out = self.sigmoid(self.conv4(out))
        return out


class DHS(nn.Module):
    """Whole structure of DHS net.
    """

    def __init__(self, block):
        super().__init__()
        self.vgg_pre = []
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Sequential(  # VGG16 net conv block1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(  # VGG16 net conv block2
            nn.MaxPool2d(2, stride=2),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(  # VGG16 net conv block3
            nn.MaxPool2d(2, stride=2),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(  # VGG16 net conv block4
            nn.MaxPool2d(2, stride=2),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(  # VGG16 net conv block5
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(14 * 14 * 512, 784)  # full connection layer
        self.layer1 = block(512)  # RCL module 1
        self.layer2 = block(256)  # RCL module 2
        self.layer3 = block(128)  # RCL module 3
        self.layer4 = block(64)  # RCL module 4
        self.features = nn.ModuleList(self.vgg_pre)

        self.__copy_param()  # copy parameters from pretrained VGG16 net

    def forward(self, x):
        """Forward and generates five predicted saliency map
         of different spatial resolution
         """
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        x = self.conv5(c4)
        bz = x.shape[0]
        x = x.view(bz, -1)
        x = self.fc(x)
        x = x.view(bz, 1, 28, -1)
        x1 = x
        x = self.layer1.forward(c4, x)
        x2 = x
        x = self.upsample(x)
        x = self.layer2.forward(c3, x)
        x3 = x
        x = self.upsample(x)
        x = self.layer3.forward(c2, x)
        x4 = x
        x = self.upsample(x)
        x = self.layer4.forward(c1, x)
        x5 = x
        return torch.sigmoid(x1), torch.sigmoid(x2), torch.sigmoid(x3), torch.sigmoid(x4), torch.sigmoid(x5)

    def __copy_param(self):
        """Load parameters from pretrained VGG16 net."""
        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        DGG_features = list(self.conv1.children())
        DGG_features.extend(list(self.conv2.children()))
        DGG_features.extend(list(self.conv3.children()))
        DGG_features.extend(list(self.conv4.children()))
        DGG_features.extend(list(self.conv5.children()))
        DGG_features = nn.Sequential(*DGG_features)

        for layer_1, layer_2 in zip(vgg16.features, DGG_features):
            if (isinstance(layer_1, nn.Conv2d) and
                    isinstance(layer_2, nn.Conv2d)):
                assert layer_1.weight.size() == layer_2.weight.size()
                assert layer_1.bias.size() == layer_2.bias.size()
                layer_2.weight.data = layer_1.weight.data
                layer_2.bias.data = layer_1.bias.data
        return


DHS_network = DHS(RCL)
