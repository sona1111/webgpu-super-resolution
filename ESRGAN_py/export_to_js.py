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

if __name__ == "__main__":
    #model_weight_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
    model_weight_path = 'models/RRDB_PSNR_x4.pth'
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_weight_path), strict=True)
    model.eval()

    # model_quant = quantize_model(model)
    # print(model_quant.state_dict()['conv_first.bias'])

    export_js_modelbin(model, 'RRDB_PSNR_x4', 'RRDB_PSNR_x4', False)
