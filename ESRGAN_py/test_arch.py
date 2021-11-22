import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import json
import sys
from PIL import Image


def make_test_image(path, size):
    im = Image.new(mode="RGB", size=(size, size),
                   color=(153, 153, 255))
    im.save(path)

torch.random.manual_seed(13)

def load_image_as_torch(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    return img_LR

def conv_fwd(inp, w, b, relu=False):
    """
    Assumes padding=3, stride=1, kernsize=3
    inp may be either one tensor, or multiple which should be stacked
    """
    if type(inp) == tuple:
        inp = np.vstack(inp)


    output = np.zeros(shape=(w.shape[0], inp.shape[1], inp.shape[2]))

    input = np.zeros(shape=(inp.shape[0], inp.shape[1], inp.shape[2]))

    kern = np.zeros(shape=(w.shape[0], w.shape[1], w.shape[2], w.shape[3]))
    bias = np.zeros(shape=(b.shape[0]))

    # load inputs to gpu main memory
    input[:, :, :] = inp[:, :, :]
    kern[:, :, :, :] = w[:, :, :, :]
    bias[:] = b[:]

    def __kernConvolve(y, x, ci, co):

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (y+i < 0 or x+j < 0 or y+i >= input.shape[1] or x+j >= input.shape[2]):
                    continue
                output[co, y, x] += (input[ci, y+i, x+j] * kern[co, ci, i+1, j+1])

    def __kernAddBias(y, x, co):
        output[co, y, x] += bias[co]

    def __kernAddBiasAndRelu(y, x, co):
        val = output[co, y, x] + bias[co]
        if val > 0:
            output[co, y, x] = val
        else:
            output[co, y, x] = 0.2*val

    def _runKern(_kern, *_args):
        for y in range(output.shape[1]):
            for x in range(output.shape[2]):
                _kern(y, x, *_args)

    for out_ch_idx in range(w.shape[0]):

        for in_ch_idx in range(inp.shape[0]):

            _runKern(__kernConvolve, in_ch_idx, out_ch_idx)

        if relu:
            _runKern(__kernAddBiasAndRelu, out_ch_idx)
        else:
            _runKern(__kernAddBias, out_ch_idx)

    return output

def rev_relu_fwd(inp1, inp2):

    # load to gpu RAM
    # output will be stored in input1
    input1 = np.zeros(shape=(inp1.shape[0], inp1.shape[1], inp1.shape[2]))
    input2 = np.zeros(shape=(inp2.shape[0], inp2.shape[1], inp2.shape[2]))

    input1[:, :, :] = inp1[:, :, :]
    input2[:, :, :] = inp2[:, :, :]

    def _runKern(_kern, *_args):
        for co in range(input1.shape[0]):
            for y in range(input1.shape[1]):
                for x in range(input1.shape[2]):
                    _kern(y, x, co, *_args)

    def __kernRevRelu(y, x, co):
        input1[co, y, x] = (input1[co, y, x] * 0.2) + input2[co, y, x]

    _runKern(__kernRevRelu)

    return input1


class ESRGAN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(ESRGAN, self).__init__()


        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_22_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_22_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_22_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_22_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_22_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_22_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_22_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_22_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_22_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_22_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_22_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_22_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_22_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_22_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_22_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)



    def forward(self, x):
        fea = self.conv_first(x)

        x_o = fea

        x1 = self.lrelu(self.RRDB_trunk_22_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_22_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_22_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_22_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_22_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o
        return x


model = ESRGAN()
model.eval()

nn.init.constant_(model.conv_first.bias, 0.01)

def esrgan(x):

    def eval_conv(name, inp, relu=False):

        _w = model.state_dict()[f'{name}.weight']
        _b = model.state_dict()[f'{name}.bias']
        _out = conv_fwd(inp, _w, _b, relu)
        return _out

    def eval_rev_relu(inp1, inp2):
        # return (inp1 * 0.2) + inp2
        _out = rev_relu_fwd(inp1, inp2)
        return _out



    fea = eval_conv('conv_first', x)

    x_o = fea

    x1 = eval_conv('RRDB_trunk_22_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_22_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_22_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_22_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_22_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)
    return x


make_test_image('test.png', 2)
img = load_image_as_torch('test.png')
#img = img[:, :1, :, :]

#img = load_image_as_torch('4x4.png')

print('image shape', img.shape)

manual_result = esrgan(img[0])
with torch.no_grad():
    torch_result = model(img).numpy()


print('------')
print('manual result shape', manual_result.shape)
print('------')
print('torch result shape', torch_result.shape)
print('------')
print('yay?', np.allclose(manual_result, torch_result[0], atol=1E-7))
