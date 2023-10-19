import torch
import torch.nn as nn


class EnhanceNet(nn.Module):
    def __init__(self, phase='train'):
        super(EnhanceNet, self).__init__()

        self.phase = phase
        dim = 64
        self.e_conv1 = nn.Conv2d(3, dim, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(dim * 2, dim, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(dim * 2, dim, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(dim, 3, 3, 1, 1, bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 2 * dim, 3, 1, 1, bias=True),
            nn.Conv2d(2 * dim, 2 * dim, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(2 * dim, dim, 3, 1, 1, bias=True),
            nn.Conv2d(dim, dim // 4, 3, 1, 1),
            nn.Conv2d(dim // 4, 3, 3, 1, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x3 = torch.cat([x2, x3], 1)
        x4 = self.relu(self.e_conv4(x3))
        x4 = torch.cat([x1, x4], 1)
        x5 = self.relu(self.e_conv5(x4))
        x6 = self.relu(self.e_conv6(x5))
        x7 = self.conv(x6)
        i = x7 + x
        i = torch.clamp(i, 0.0001, 1)

        r = x / i
        r = torch.clamp(r, 0, 1)

        if self.phase == 'train':
            return r, i
        else:
            return r



if __name__ == '__main__':
    net = EnhanceNet()
    input = torch.rand(2, 3, 384, 384)
    output = net(input)
    print(output[0].shape)
    # for p in net.state_dict():
    #     print(p)



