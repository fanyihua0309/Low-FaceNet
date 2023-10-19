import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = vgg16(pretrained=True).features.cuda()
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h

        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3)
        return out


class ContrastLoss(nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()
        # self.vgg = Vgg19().cuda()
        self.vgg = Vgg16().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            d_an = self.l1(a_vgg[i], n_vgg[i].detach())
            contrastive = d_ap / (d_an + 1e-7)
            loss += self.weights[i] * contrastive
        return loss


class ContrastBrightnessLoss(nn.Module):
    def __init__(self):
        super(ContrastBrightnessLoss, self).__init__()

    def forward(self, a, p, n):
        bright_a = torch.mean(a, 1, keepdim=True)
        bright_p = torch.mean(p, 1, keepdim=True)
        bright_n = torch.mean(n, 1, keepdim=True)
        d_ap = torch.mean(abs(bright_a - bright_p))
        d_an = torch.mean(abs(bright_a - bright_n))
        loss = d_ap / (d_an + 1e-7)
        # print(f'd_ap: {d_ap} d_an: {d_an}')
        return loss