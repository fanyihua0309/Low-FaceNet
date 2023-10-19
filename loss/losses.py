from torchvision.models.vgg import vgg16
import torch
import torch.nn as nn


# 颜色一致性损失
class ColorConsistencyLoss(nn.Module):
    def __init__(self):
        super(ColorConsistencyLoss, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, dim=(2, 3), keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k


# 感知损失
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        features = vgg16(pretrained=True).features.cuda()

        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x, y):
        h = self.to_relu_1_2(x)
        h = self.to_relu_2_2(h)
        h = self.to_relu_3_3(h)
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        g = self.to_relu_1_2(y)
        g = self.to_relu_2_2(g)
        g = self.to_relu_3_3(g)
        g = self.to_relu_4_3(g)
        g_relu_4_3 = g
        content_loss = self.mse_loss(
            h_relu_4_3, g_relu_4_3)
        return content_loss


# 语义亮度一致性损失
class SegBrightnessLoss(nn.Module):
    def __init__(self):
        super(SegBrightnessLoss, self).__init__()

    def forward(self, x, y):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        a1 = torch.zeros(b, h, w).cuda()
        d = 0
        # 语义分割类别数
        for i in range(11):
            a2 = torch.where(y == i, x, a1).cuda()
            d2 = torch.mean(a2)
            a2 = torch.where(a2 == 0, d2, a2).cuda()
            d3 = torch.mean(torch.pow(a2 - torch.FloatTensor([d2]).cuda(), 2))
            d = d + d3
        return d
