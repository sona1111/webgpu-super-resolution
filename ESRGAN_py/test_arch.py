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
from test_arch_torch import ESRGAN
from collections import OrderedDict

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

# def vstack(mats):
#     total_first_dim = 0
#     for mat in mats:
#         total_first_dim += mat.shape[0]
#     new_shape = (total_first_dim, mats[0].shape[1], mats[0].shape[2])
#     new_mat = np.empty(new_shape)
#     first_dim = 0
#     for mat in mats:
#         new_mat[first_dim:first_dim+mat.shape[0], :, :] =

def conv_fwd(inp, w, b, relu=False):
    """
    Assumes padding=3, stride=1, kernsize=3
    inp may be either one tensor, or multiple which should be stacked
    """
    if type(inp) == tuple:
        print(inp[0].shape)
        inp = np.vstack(inp)
        print(inp.shape)
        assert False


    output = np.zeros(shape=(w.shape[0], inp.shape[1], inp.shape[2]))

    input = np.zeros(shape=(inp.shape[0], inp.shape[1], inp.shape[2]))

    weight = np.zeros(shape=(w.shape[0], w.shape[1], w.shape[2], w.shape[3]))
    bias = np.zeros(shape=(b.shape[0]))

    # load inputs to gpu main memory
    input[:, :, :] = inp[:, :, :]
    weight[:, :, :, :] = w[:, :, :, :]
    bias[:] = b[:]

    def __kernConvolve(y, x, ci, co):

        dbg_arr = []

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:





                if (y+i < 0 or x+j < 0 or y+i >= input.shape[1] or x+j >= input.shape[2]):
                    dbg_arr.append(0)
                    continue
                kern_idx = (ci * weight.shape[1] * weight.shape[2] * weight.shape[3]) + \
                           (co * weight.shape[2] * weight.shape[3]) + \
                           ((i+1) * weight.shape[3]) + (j+1)
                dbg_arr.append(input[ci, y+i, x+j])
                output[co, y, x] += (input[ci, y+i, x+j] * weight[co, ci, i+1, j+1])


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


    print(output[0:2])
    assert False
    return output

def rev_relu_fwd(inp1, inp2, add_only=False):

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

    def __kernRevReluAddOnly(y, x, co):
        input1[co, y, x] = (input1[co, y, x]) + input2[co, y, x]

    if add_only:
        _runKern(__kernRevReluAddOnly)
    else:
        _runKern(__kernRevRelu)

    return input1

def interpolate_fwd(inp):

    # load to gpu RAM
    input = np.zeros(shape=(inp.shape[0], inp.shape[1], inp.shape[2]))
    output = np.zeros(shape=(inp.shape[0], inp.shape[1]*2, inp.shape[2]*2))

    input[:, :, :] = inp[:, :, :]

    def _runKern(_kern, *_args):
        for co in range(input.shape[0]):
            for y in range(input.shape[1]):
                for x in range(input.shape[2]):
                    _kern(y, x, co, *_args)

    def __kernInterp(y, x, co):
        output[co, y*2, x*2] = input[co, y, x]
        output[co, (y*2)+1, x*2] = input[co, y, x]
        output[co, y*2, (x*2)+1] = input[co, y, x]
        output[co, (y*2)+1, (x*2)+1] = input[co, y, x]

    _runKern(__kernInterp)
    return output

model = ESRGAN()
model.eval()

if False:
    nn.init.constant_(model.conv_first.bias, 0.01)
else:
    state_dict = torch.load('models/RRDB_ESRGAN_x4.pth')

    def change_key(old, new, od):
        d2 = OrderedDict([(new, v) if k == old else (k, v) for k, v in od.items()])
        return d2

    def dots_to_underscores(_new_model_state):
        new_model_state = _new_model_state
        keys = list(new_model_state.keys())
        for key in keys:
            if key.startswith('RRDB_trunk'):
                parts = key.split('.')
                new_key = '_'.join(parts[:-1]) + '.' + parts[-1]
                new_model_state = change_key(key, new_key, new_model_state)
        return new_model_state

    new_state = dots_to_underscores(state_dict)

    model.load_state_dict(new_state, strict=True)

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
    print(x_o.shape)
    assert False

    #----------------
    x1 = eval_conv('RRDB_trunk_0_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_0_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_0_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_0_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_0_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_0_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_0_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_0_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_0_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_0_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_0_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_0_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_0_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_0_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_0_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_1_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_1_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_1_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_1_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_1_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_1_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_1_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_1_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_1_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_1_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_1_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_1_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_1_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_1_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_1_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------

    #----------------
    x1 = eval_conv('RRDB_trunk_2_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_2_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_2_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_2_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_2_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_2_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_2_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_2_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_2_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_2_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_2_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_2_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_2_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_2_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_2_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_3_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_3_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_3_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_3_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_3_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_3_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_3_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_3_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_3_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_3_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_3_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_3_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_3_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_3_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_3_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_4_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_4_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_4_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_4_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_4_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_4_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_4_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_4_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_4_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_4_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_4_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_4_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_4_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_4_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_4_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_5_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_5_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_5_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_5_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_5_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_5_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_5_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_5_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_5_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_5_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_5_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_5_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_5_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_5_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_5_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_6_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_6_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_6_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_6_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_6_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_6_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_6_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_6_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_6_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_6_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_6_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_6_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_6_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_6_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_6_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_7_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_7_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_7_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_7_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_7_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_7_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_7_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_7_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_7_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_7_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_7_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_7_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_7_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_7_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_7_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_8_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_8_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_8_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_8_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_8_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_8_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_8_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_8_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_8_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_8_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_8_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_8_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_8_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_8_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_8_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_9_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_9_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_9_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_9_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_9_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_9_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_9_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_9_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_9_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_9_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_9_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_9_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_9_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_9_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_9_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_10_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_10_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_10_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_10_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_10_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_10_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_10_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_10_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_10_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_10_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_10_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_10_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_10_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_10_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_10_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------

    #----------------
    x1 = eval_conv('RRDB_trunk_11_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_11_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_11_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_11_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_11_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_11_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_11_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_11_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_11_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_11_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_11_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_11_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_11_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_11_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_11_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------

    #----------------
    x1 = eval_conv('RRDB_trunk_12_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_12_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_12_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_12_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_12_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_12_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_12_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_12_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_12_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_12_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_12_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_12_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_12_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_12_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_12_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_13_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_13_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_13_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_13_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_13_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_13_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_13_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_13_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_13_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_13_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_13_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_13_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_13_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_13_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_13_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------

    #----------------
    x1 = eval_conv('RRDB_trunk_14_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_14_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_14_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_14_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_14_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_14_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_14_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_14_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_14_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_14_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_14_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_14_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_14_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_14_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_14_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_15_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_15_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_15_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_15_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_15_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_15_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_15_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_15_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_15_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_15_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_15_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_15_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_15_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_15_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_15_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_16_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_16_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_16_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_16_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_16_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_16_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_16_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_16_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_16_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_16_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_16_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_16_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_16_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_16_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_16_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_17_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_17_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_17_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_17_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_17_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_17_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_17_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_17_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_17_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_17_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_17_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_17_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_17_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_17_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_17_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_18_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_18_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_18_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_18_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_18_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_18_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_18_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_18_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_18_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_18_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_18_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_18_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_18_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_18_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_18_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_19_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_19_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_19_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_19_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_19_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_19_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_19_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_19_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_19_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_19_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_19_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_19_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_19_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_19_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_19_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_20_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_20_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_20_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_20_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_20_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_20_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_20_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_20_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_20_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_20_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_20_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_20_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_20_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_20_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_20_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    #----------------
    x1 = eval_conv('RRDB_trunk_21_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_21_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_21_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_21_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_21_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_21_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_21_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_21_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_21_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_21_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_21_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_21_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_21_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_21_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_21_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------

    #----------------
    x1 = eval_conv('RRDB_trunk_22_RDB1_conv1', x_o, relu=True)
    x2 = eval_conv('RRDB_trunk_22_RDB1_conv2', (x_o, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_22_RDB1_conv3', (x_o, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_22_RDB1_conv4', (x_o, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_22_RDB1_conv5', (x_o, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x_o)

    x1 = eval_conv('RRDB_trunk_22_RDB2_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_22_RDB2_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_22_RDB2_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_22_RDB2_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_22_RDB2_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)

    x1 = eval_conv('RRDB_trunk_22_RDB3_conv1', x, relu=True)
    x2 = eval_conv('RRDB_trunk_22_RDB3_conv2', (x, x1), relu=True)
    x3 = eval_conv('RRDB_trunk_22_RDB3_conv3', (x, x1, x2), relu=True)
    x4 = eval_conv('RRDB_trunk_22_RDB3_conv4', (x, x1, x2, x3), relu=True)
    x5 = eval_conv('RRDB_trunk_22_RDB3_conv5', (x, x1, x2, x3, x4), relu=False)
    x = eval_rev_relu(x5, x)
    x_o = eval_rev_relu(x, x_o)
    #----------------


    trunk = eval_conv('trunk_conv', x_o, relu=False)

    fea = rev_relu_fwd(fea, trunk, add_only=True)


    fea = eval_conv('upconv1', interpolate_fwd(fea), relu=True)
    fea = eval_conv('upconv2', interpolate_fwd(fea), relu=True)
    out = eval_conv('conv_last', eval_conv('HRconv', fea, relu=True), relu=False)

    return out


make_test_image('test.png', 2)
img = load_image_as_torch('test.png')
#img = img[:, :1, :, :]

img = load_image_as_torch('../3x2.png')

print('image shape', img.shape)

manual_result = esrgan(img[0])
with torch.no_grad():
    torch_result = model(img).numpy()
    print('img', img)
    print('res', torch_result)
    print('man', manual_result)



print('------')
print('manual result shape', manual_result.shape)
print('------')
print('torch result shape', torch_result.shape)
print('------')
print('yay?', np.allclose(manual_result, torch_result[0], atol=1E-5))
