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

def export_js_modelbin(model, path):

    if not os.path.exists(path):
        os.makedirs(path)

    modelinfo = {'arch': 'ESRGAN',
                 'layers':[

                 ]}

    for name in model.state_dict():
        if name.endswith('weight'):
            continue

        modelinfo['layers'].append(name.replace('.bias', ''))


    with open(os.path.join(path, 'modelinfo.json'), 'w') as f:
        json.dump(modelinfo, f)

    for param_tensor in model.state_dict():
        with open(os.path.join(path, f'{param_tensor}.bin'), 'wb') as f:
            f.write(model.state_dict()[param_tensor].numpy().tobytes())



if __name__ == "__main__":
    model_weight_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_weight_path), strict=True)
    model.eval()

    export_js_modelbin(model, 'RRDB_ESRGAN_x4')
