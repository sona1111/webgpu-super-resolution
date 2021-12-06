import os
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
import torch.nn as nn
import json
import sys
import cv2
from PIL import Image
from test_arch_torch import ESRGAN
from collections import OrderedDict
import shutil

def load_image_as_torch(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    return img_LR

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

model = ESRGAN()
model.eval()
model.load_state_dict(new_state, strict=True)

img = load_image_as_torch('../baboon.png')
with torch.no_grad():
    torch_result = model(img).numpy()


def grid_exp(net_out, path):

    # if os.path.exists(folder):
    #     shutil.rmtree(folder)
    # os.makedirs(folder)



    xSize = net_out[0, 0].shape[0]
    ySize = net_out[0, 0].shape[1]
    new_im = Image.new('L', (8*xSize,8*ySize))
    x = 0
    y = 0

    for i, image_data in enumerate(net_out[0]):
        #image_data = np.transpose(image_data, )

        #print(image_data.shape)


        # output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        image_data = (image_data * 255.0).round()
        im = Image.fromarray(image_data).convert('L')
        #im.save(f'{folder}/test_ch_{i}.png')

        x += xSize
        if x > (xSize * 8):
            x = 0
            y += ySize

        new_im.paste(im, (x,y))

    new_im.save(path)
    # files = [ f for f in listdir("/mnt/hgfs/Documents/Notebooks/test1/") if isfile(join("/mnt/hgfs/Documents/Notebooks/test1/", f)) ]
    #

    #
    # index = 0
    # for i in xrange(0,3000,300):
    #     for j in xrange(0,3000,300):
    #         im = Image.open(files[index])
    #         im.thumbnail((300,300))
    #         new_im.paste(im, (i,j))
    #         index += 1
    #
    # new_im.save("hola.png")

grid_exp(torch_result, 'intermediate/trunk_and_rrbd.png')

