import torch
import torch.nn as nn


class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.sigma = 0.1

    def rgb2yCbCr(self, input_im):
        im_flat = input_im.contiguous().view(-1, 3).float()
        mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).cuda()
        bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda()
        temp = im_flat.mm(mat) + bias
        out = temp.view(input_im.shape[0], 3, input_im.shape[2], input_im.shape[3])
        return out

    def norm(self, tensor, p):
        return torch.mean(torch.pow(torch.abs(tensor), p))

    # output: output      input: input
    def forward(self, input, output):
        self.output = output
        self.input = self.rgb2yCbCr(input)
        self.input = input
        sigma_color = -1.0 / 2 * self.sigma * self.sigma
        w1 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :] - self.input[:, :, :-1, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w2 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :] - self.input[:, :, 1:, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w3 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 1:] - self.input[:, :, :, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w4 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-1] - self.input[:, :, :, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w5 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-1] - self.input[:, :, 1:, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w6 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 1:] - self.input[:, :, :-1, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w7 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-1] - self.input[:, :, :-1, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w8 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 1:] - self.input[:, :, 1:, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w9 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :] - self.input[:, :, :-2, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w10 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :] - self.input[:, :, 2:, :], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w11 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 2:] - self.input[:, :, :, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w12 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-2] - self.input[:, :, :, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w13 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-1] - self.input[:, :, 2:, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w14 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 1:] - self.input[:, :, :-2, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w15 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-1] - self.input[:, :, :-2, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w16 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 1:] - self.input[:, :, 2:, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w17 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-2] - self.input[:, :, 1:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w18 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 2:] - self.input[:, :, :-1, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w19 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-2] - self.input[:, :, :-1, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w20 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 2:] - self.input[:, :, 1:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w21 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-2] - self.input[:, :, 2:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w22 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 2:] - self.input[:, :, :-2, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w23 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-2] - self.input[:, :, :-2, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w24 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 2:] - self.input[:, :, 2:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        p = 1.0

        pixel_grad1 = w1 * self.norm((self.output[:, :, 1:, :] - self.output[:, :, :-1, :]), p)
        pixel_grad2 = w2 * self.norm((self.output[:, :, :-1, :] - self.output[:, :, 1:, :]), p)
        pixel_grad3 = w3 * self.norm((self.output[:, :, :, 1:] - self.output[:, :, :, :-1]), p)
        pixel_grad4 = w4 * self.norm((self.output[:, :, :, :-1] - self.output[:, :, :, 1:]), p)
        pixel_grad5 = w5 * self.norm((self.output[:, :, :-1, :-1] - self.output[:, :, 1:, 1:]), p)
        pixel_grad6 = w6 * self.norm((self.output[:, :, 1:, 1:] - self.output[:, :, :-1, :-1]), p)
        pixel_grad7 = w7 * self.norm((self.output[:, :, 1:, :-1] - self.output[:, :, :-1, 1:]), p)
        pixel_grad8 = w8 * self.norm((self.output[:, :, :-1, 1:] - self.output[:, :, 1:, :-1]), p)
        pixel_grad9 = w9 * self.norm((self.output[:, :, 2:, :] - self.output[:, :, :-2, :]), p)
        pixel_grad10 = w10 * self.norm((self.output[:, :, :-2, :] - self.output[:, :, 2:, :]), p)
        pixel_grad11 = w11 * self.norm((self.output[:, :, :, 2:] - self.output[:, :, :, :-2]), p)
        pixel_grad12 = w12 * self.norm((self.output[:, :, :, :-2] - self.output[:, :, :, 2:]), p)
        pixel_grad13 = w13 * self.norm((self.output[:, :, :-2, :-1] - self.output[:, :, 2:, 1:]), p)
        pixel_grad14 = w14 * self.norm((self.output[:, :, 2:, 1:] - self.output[:, :, :-2, :-1]), p)
        pixel_grad15 = w15 * self.norm((self.output[:, :, 2:, :-1] - self.output[:, :, :-2, 1:]), p)
        pixel_grad16 = w16 * self.norm((self.output[:, :, :-2, 1:] - self.output[:, :, 2:, :-1]), p)
        pixel_grad17 = w17 * self.norm((self.output[:, :, :-1, :-2] - self.output[:, :, 1:, 2:]), p)
        pixel_grad18 = w18 * self.norm((self.output[:, :, 1:, 2:] - self.output[:, :, :-1, :-2]), p)
        pixel_grad19 = w19 * self.norm((self.output[:, :, 1:, :-2] - self.output[:, :, :-1, 2:]), p)
        pixel_grad20 = w20 * self.norm((self.output[:, :, :-1, 2:] - self.output[:, :, 1:, :-2]), p)
        pixel_grad21 = w21 * self.norm((self.output[:, :, :-2, :-2] - self.output[:, :, 2:, 2:]), p)
        pixel_grad22 = w22 * self.norm((self.output[:, :, 2:, 2:] - self.output[:, :, :-2, :-2]), p)
        pixel_grad23 = w23 * self.norm((self.output[:, :, 2:, :-2] - self.output[:, :, :-2, 2:]), p)
        pixel_grad24 = w24 * self.norm((self.output[:, :, :-2, 2:] - self.output[:, :, 2:, :-2]), p)

        ReguTerm1 = torch.mean(pixel_grad1) \
                    + torch.mean(pixel_grad2) \
                    + torch.mean(pixel_grad3) \
                    + torch.mean(pixel_grad4) \
                    + torch.mean(pixel_grad5) \
                    + torch.mean(pixel_grad6) \
                    + torch.mean(pixel_grad7) \
                    + torch.mean(pixel_grad8) \
                    + torch.mean(pixel_grad9) \
                    + torch.mean(pixel_grad10) \
                    + torch.mean(pixel_grad11) \
                    + torch.mean(pixel_grad12) \
                    + torch.mean(pixel_grad13) \
                    + torch.mean(pixel_grad14) \
                    + torch.mean(pixel_grad15) \
                    + torch.mean(pixel_grad16) \
                    + torch.mean(pixel_grad17) \
                    + torch.mean(pixel_grad18) \
                    + torch.mean(pixel_grad19) \
                    + torch.mean(pixel_grad20) \
                    + torch.mean(pixel_grad21) \
                    + torch.mean(pixel_grad22) \
                    + torch.mean(pixel_grad23) \
                    + torch.mean(pixel_grad24)
        total_term = ReguTerm1
        return total_term