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

def get_conv_fwd_buf_sizes(inp, w):
    if type(inp) == tuple:

        inp = np.vstack(inp)
    output_shape = (w.shape[0], inp.shape[1], inp.shape[2])
    input_shape = (inp.shape[0], inp.shape[1], inp.shape[2])
    return input_shape, output_shape

buffers = {}

def conv_fwd(inp, w, b, relu=False):
    """
    Assumes padding=3, stride=1, kernsize=3
    inp may be either one tensor, or multiple which should be stacked
    """
    if type(inp) == tuple:

        inp = np.vstack(inp)


    output = np.zeros(shape=(w.shape[0], inp.shape[1], inp.shape[2]))

    input = np.zeros(shape=(inp.shape[0], inp.shape[1], inp.shape[2]))

    weight = np.zeros(shape=(w.shape[0], w.shape[1], w.shape[2], w.shape[3]))
    bias = np.zeros(shape=(b.shape[0]))

    # load inputs to gpu main memory
    input[:, :, :] = inp[:, :, :]
    weight[:, :, :, :] = w[:, :, :, :]
    bias[:] = b[:]

    def __kernConvolve(y, x, ci, co):



        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:

                if (y+i < 0 or x+j < 0 or y+i >= input.shape[1] or x+j >= input.shape[2]):
                    continue

                output[co, y, x] += (input[ci, y+i, x+j] * weight[co, ci, i+1, j+1])




    def __kernAddBias(y, x, co):
        output[co, y, x] += bias[co]

    def __kernAddBiasAndRelu(y, x, co):
        val = output[co, y, x] + bias[co]
        if val > 0:
            output[co, y, x] = val
        else:
            output[co, y, x] = 0.2*val

    def __kernConvolveAndAddBias(y, x, ci, num_co):

        outputs = []
        for co in range(num_co):

            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:


                    if (y+i < 0 or x+j < 0 or y+i >= input.shape[1] or x+j >= input.shape[2]):
                        continue




                    # if ci == 0 and co == 1 and x == 1 and y == 1:
                    output[co, y, x] += (input[ci, y+i, x+j] * weight[co, ci, i+1, j+1])

            if ci == 0:
                output[co, y, x] += bias[co]

            # if ci == 0 and co == 1 and x == 1 and y == 1:
            #     outputs.append(output[co, y, x])
            #
            # if ci == 0 and co == 1 and x == 1 and y == 1:
            #     print(outputs)
            #     assert False



    def __kernLRELU(y, x, co):
        val = output[co, y, x]
        if val <= 0:
            output[co, y, x] = 0.2*val


    def _runKern2d(_kern, *_args):
        for y in range(output.shape[1]):
            for x in range(output.shape[2]):
                _kern(y, x, *_args)

    def _runKern3d(_kern, zrange, *_args):
        for z in range(zrange):
            for y in range(output.shape[1]):
                for x in range(output.shape[2]):
                    _kern(y, x, z, *_args)

    def _runKern4d(_kern, *_args):

            for in_ch_idx in range(inp.shape[0]):
                for y in range(output.shape[1]):
                    for x in range(output.shape[2]):
                        _kern(y, x, in_ch_idx, num_out_ch, *_args)


    _runKern3d(__kernConvolveAndAddBias, inp.shape[0], w.shape[0])
    if relu:
        _runKern3d(__kernLRELU, w.shape[0])


    # for out_ch_idx in range(w.shape[0]):
    #
    #
    #
    #     _runKern(__kernConvolve, out_ch_idx)
    #
    #
    #     if relu:
    #         _runKern2d(__kernAddBiasAndRelu, out_ch_idx)
    #     else:
    #         _runKern2d(__kernAddBias, out_ch_idx)


    return output

def rev_relu_fwd(inp1, inp2, add_only=False):

    print('eval relu fwd', (inp1.shape, inp2.shape))

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

    """
    Strategy for V2

    -load _all_ weight+bias matrix buffers first
    -prepare multiple (seven?) output buffers to store all intermediate data
    -for each step we just write to the next output buffer instead of reading back to CPU

    conv_first: 3 * w * h ; 64 * w * h

    rrdb_in: 3 * w * h ; 64 * w * h
    rdb_in: 3 * w * h ; 64 * w * h
    x1: 64 * w * h ; 32 * w * h
    x2: 96 * w * h ; 32 * w * h
    x3: 128 * w * h ; 32 * w * h
    x4: 160 * w * h ; 32 * w * h
    x5: 192 * w * h ; 64 * w * h

    trunk_conv: 64 * w * h ; 64 * w * h
    upconv1: 64 * w*2 * h*2 ; 64 * w*2 * h*2
    upconv2: 64 * w*4 * h*4 ; 64 * w*4 * h*4
    HRconv: 64 * w*4 * h*4 ; 64 * w*4 * h*4
    conv_last: 64 * w*4 * h*4 ; 3 * w*4 * h*4

    total space without repeating:
    image 2x3 = 4 bytes * 41046 = 0.164MB
    image 1000x1000 = 4 bytes * 6841000000 = 27GB!
    image 2000x2000 = 4 bytes * 27364000000 = 109GB!

    first allocate conv_first and the seven RRDB buffers to get to the

    """

    def eval_conv(name, inp, relu=False):




        _w = model.state_dict()[f'{name}.weight']
        _b = model.state_dict()[f'{name}.bias']

        print('eval conv', name, get_conv_fwd_buf_sizes(inp, _w))

        _out = conv_fwd(inp, _w, _b, relu)
        return _out

    def eval_rev_relu(inp1, inp2):
        # return (inp1 * 0.2) + inp2
        #print('eval relu', (inp1.shape, inp2.shape))

        _out = rev_relu_fwd(inp1, inp2)
        return _out

    fea = eval_conv('conv_first', x, relu=False)


    rrdb_in = fea

    for i in range(23): # 23
        rdb_in = rrdb_in
        #----------------
        for j in range(1,4):

            # create two buffer of (64+192) called rdb_in
            #
            # conv from rdb_in (0-64) to rbd_in (65+=32)
            # conv from rdb_in (0-64+32) to rdb_in (64+32+=32)
            # ...
            # conv from rdb_in (full) to rdb_in2 (0-64)
            # flip


            x1 = eval_conv(f'RRDB_trunk_{i}_RDB{j}_conv1', rdb_in, relu=True)
            #return rdb_in
            x2 = eval_conv(f'RRDB_trunk_{i}_RDB{j}_conv2', (rdb_in, x1), relu=True)


            x3 = eval_conv(f'RRDB_trunk_{i}_RDB{j}_conv3', (rdb_in, x1, x2), relu=True)
            x4 = eval_conv(f'RRDB_trunk_{i}_RDB{j}_conv4', (rdb_in, x1, x2, x3), relu=True)
            x5 = eval_conv(f'RRDB_trunk_{i}_RDB{j}_conv5', (rdb_in, x1, x2, x3, x4), relu=False)
            #return x5
            rdb_in = eval_rev_relu(x5, rdb_in)
            break
            #return rdb_in


        #return rrdb_in
        rrdb_in = eval_rev_relu(rdb_in, rrdb_in)
        break
        #----------------


    # trunk_conv result will fit in rdbbuf (192) prealloc, read direct from rrdb_in

    trunk = eval_conv('trunk_conv', rrdb_in, relu=False)

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
    # print('img', img)
    # print('res', torch_result)
    print('man', manual_result[0])



print('------')
print('manual result shape', manual_result.shape)
print('------')
print('torch result shape', torch_result.shape)
print('------')
print('yay?', np.allclose(manual_result, torch_result[0], atol=1E-5))
