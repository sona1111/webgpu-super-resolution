import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import torch.nn as nn
import json
import sys
import re
from collections import OrderedDict

def export_js_modelbin(model, path, arch, is_quantized):

    if not os.path.exists(path):
        os.makedirs(path)

    modelinfo = {'arch': arch,
                 'is_quantized': is_quantized,
                 'layers':[

                 ]}

    cur_layer = {}
    for name in model.state_dict():
        l = model.state_dict()[name]
        if name.endswith('weight'):
            cur_layer['name'] = name.replace('.weight', '')
            cur_layer['wshape'] = list(l.shape)
            continue

        cur_layer['bshape'] = list(l.shape)
        modelinfo['layers'].append(cur_layer)
        cur_layer = {}


    with open(os.path.join(path, 'modelinfo.json'), 'w') as f:
        json.dump(modelinfo, f)

    for param_tensor in model.state_dict():
        print('exporting', param_tensor)
        with open(os.path.join(path, f'{param_tensor}.bin'), 'wb') as f:
            raw = model.state_dict()[param_tensor]
            if is_quantized:
                raw = raw.int_repr()
            f.write(raw.numpy().tobytes())

def quantize_model(model):

    backend = "qnnpack"
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_static_quantized = torch.quantization.prepare(model, inplace=False)
    model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
    return model_static_quantized

def esrgan_plus_to_esrgan(states):

    tr_keys = {
        'conv_hr.weight': 'HRconv.weight',
        'conv_hr.bias': 'HRconv.bias',
        'conv_up2.weight': 'upconv2.weight',
        'conv_up2.bias': 'upconv2.bias',
        'conv_up1.weight': 'upconv1.weight',
        'conv_up1.bias': 'upconv1.bias',
        'conv_body.weight': 'trunk_conv.weight',
        'conv_body.bias': 'trunk_conv.bias',
    }
    states = states['params_ema']
    new_states = OrderedDict()
    keys = states.keys()
    for key in keys:

        match = re.search(r'body\.(\d+)\.rdb(\d+)\.conv(\d+)\.(\w+)', key)
        if match:
            key_new = f'RRDB_trunk.{match.groups()[0]}.RDB{match.groups()[1]}.conv{match.groups()[2]}.{match.groups()[3]}'
            #print(match.groups())
        elif key in tr_keys:
            key_new = tr_keys[key]
        else:
            key_new = key

        new_states[key_new] = states[key]
    return new_states

if __name__ == "__main__":
    #model_weight_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth

    for name in ['RealESRNet_x4plus',  'RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B']: # ,: 'RealESRGAN_x2plus'
        print(name)
        model_weight_path = f'models/{name}.pth'

        states = torch.load(model_weight_path)

        if True:
            states = esrgan_plus_to_esrgan(states)
        #print(list(states['params_ema'].keys()))
        #assert False

        if name == 'RealESRGAN_x4plus_anime_6B':
            model = arch.RRDBNet(3, 3, 64, 6, gc=32)
        else:
            model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        model.load_state_dict(states, strict=True)
        model.eval()

        # model_quant = quantize_model(model)
        # print(model_quant.state_dict()['conv_first.bias'])

        export_js_modelbin(model, name, name, False)
