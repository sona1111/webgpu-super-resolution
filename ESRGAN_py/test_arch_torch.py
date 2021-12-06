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
import torch.nn.functional as F


#
# class ESRGAN(nn.Module):
#     def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
#         super(ESRGAN, self).__init__()
#
#
#         self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
#         self.RRDB_trunk_22_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
#         self.RRDB_trunk_22_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
#         self.RRDB_trunk_22_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
#         self.RRDB_trunk_22_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
#         self.RRDB_trunk_22_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
#         self.RRDB_trunk_22_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
#         self.RRDB_trunk_22_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
#         self.RRDB_trunk_22_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
#         self.RRDB_trunk_22_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
#         self.RRDB_trunk_22_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
#         self.RRDB_trunk_22_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
#         self.RRDB_trunk_22_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
#         self.RRDB_trunk_22_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
#         self.RRDB_trunk_22_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
#         self.RRDB_trunk_22_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#
#
#
#     def forward(self, x):
#         fea = self.conv_first(x)
#
#         x_o = fea
#
#         x1 = self.lrelu(self.RRDB_trunk_22_RDB1_conv1(x_o))
#         x2 = self.lrelu(self.RRDB_trunk_22_RDB1_conv2(torch.cat((x_o, x1), 1)))
#         x3 = self.lrelu(self.RRDB_trunk_22_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
#         x4 = self.lrelu(self.RRDB_trunk_22_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
#         x5 = self.RRDB_trunk_22_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
#         x = x5 * 0.2 + x_o
#         return x


class ESRGAN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(ESRGAN, self).__init__()



        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_0_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_0_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_0_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_0_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_0_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_0_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_0_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_0_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_0_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_0_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_0_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_0_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_0_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_0_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_0_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_1_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_1_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_1_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_1_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_1_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_1_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_1_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_1_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_1_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_1_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_1_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_1_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_1_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_1_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_1_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_2_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_2_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_2_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_2_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_2_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_2_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_2_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_2_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_2_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_2_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_2_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_2_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_2_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_2_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_2_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_3_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_3_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_3_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_3_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_3_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_3_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_3_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_3_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_3_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_3_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_3_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_3_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_3_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_3_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_3_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_4_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_4_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_4_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_4_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_4_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_4_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_4_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_4_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_4_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_4_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_4_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_4_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_4_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_4_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_4_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_5_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_5_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_5_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_5_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_5_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_5_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_5_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_5_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_5_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_5_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_5_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_5_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_5_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_5_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_5_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_6_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_6_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_6_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_6_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_6_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_6_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_6_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_6_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_6_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_6_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_6_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_6_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_6_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_6_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_6_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_7_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_7_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_7_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_7_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_7_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_7_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_7_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_7_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_7_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_7_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_7_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_7_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_7_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_7_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_7_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_8_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_8_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_8_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_8_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_8_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_8_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_8_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_8_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_8_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_8_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_8_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_8_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_8_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_8_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_8_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_9_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_9_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_9_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_9_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_9_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_9_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_9_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_9_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_9_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_9_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_9_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_9_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_9_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_9_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_9_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_10_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_10_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_10_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_10_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_10_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_10_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_10_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_10_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_10_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_10_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_10_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_10_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_10_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_10_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_10_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_11_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_11_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_11_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_11_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_11_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_11_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_11_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_11_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_11_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_11_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_11_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_11_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_11_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_11_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_11_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_12_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_12_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_12_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_12_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_12_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_12_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_12_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_12_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_12_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_12_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_12_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_12_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_12_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_12_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_12_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_13_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_13_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_13_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_13_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_13_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_13_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_13_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_13_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_13_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_13_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_13_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_13_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_13_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_13_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_13_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_14_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_14_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_14_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_14_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_14_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_14_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_14_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_14_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_14_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_14_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_14_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_14_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_14_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_14_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_14_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_15_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_15_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_15_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_15_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_15_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_15_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_15_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_15_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_15_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_15_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_15_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_15_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_15_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_15_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_15_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_16_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_16_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_16_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_16_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_16_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_16_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_16_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_16_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_16_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_16_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_16_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_16_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_16_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_16_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_16_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_17_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_17_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_17_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_17_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_17_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_17_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_17_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_17_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_17_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_17_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_17_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_17_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_17_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_17_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_17_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_18_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_18_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_18_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_18_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_18_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_18_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_18_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_18_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_18_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_18_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_18_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_18_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_18_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_18_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_18_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_19_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_19_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_19_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_19_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_19_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_19_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_19_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_19_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_19_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_19_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_19_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_19_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_19_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_19_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_19_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_20_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_20_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_20_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_20_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_20_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_20_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_20_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_20_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_20_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_20_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_20_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_20_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_20_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_20_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_20_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk_21_RDB1_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_21_RDB1_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_21_RDB1_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_21_RDB1_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_21_RDB1_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_21_RDB2_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_21_RDB2_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_21_RDB2_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_21_RDB2_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_21_RDB2_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk_21_RDB3_conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_21_RDB3_conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_21_RDB3_conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_21_RDB3_conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.RRDB_trunk_21_RDB3_conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)

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

        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)



    def forward(self, x):

        fea = self.conv_first(x)
        #return fea

        x_o = fea

        #----------
        x1 = self.lrelu(self.RRDB_trunk_0_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_0_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_0_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_0_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_0_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o



        x1 = self.lrelu(self.RRDB_trunk_0_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_0_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_0_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_0_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_0_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_0_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_0_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_0_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_0_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_0_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------
        #return x_o

        #----------
        x1 = self.lrelu(self.RRDB_trunk_1_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_1_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_1_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_1_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_1_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_1_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_1_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_1_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_1_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_1_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_1_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_1_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_1_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_1_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_1_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------

        #----------
        x1 = self.lrelu(self.RRDB_trunk_2_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_2_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_2_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_2_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_2_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_2_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_2_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_2_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_2_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_2_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_2_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_2_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_2_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_2_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_2_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_3_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_3_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_3_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_3_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_3_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_3_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_3_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_3_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_3_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_3_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_3_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_3_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_3_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_3_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_3_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_4_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_4_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_4_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_4_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_4_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_4_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_4_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_4_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_4_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_4_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_4_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_4_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_4_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_4_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_4_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_5_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_5_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_5_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_5_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_5_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_5_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_5_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_5_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_5_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_5_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_5_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_5_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_5_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_5_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_5_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_6_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_6_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_6_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_6_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_6_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_6_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_6_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_6_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_6_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_6_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_6_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_6_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_6_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_6_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_6_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_7_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_7_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_7_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_7_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_7_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_7_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_7_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_7_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_7_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_7_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_7_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_7_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_7_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_7_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_7_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_8_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_8_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_8_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_8_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_8_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_8_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_8_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_8_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_8_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_8_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_8_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_8_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_8_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_8_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_8_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_9_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_9_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_9_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_9_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_9_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_9_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_9_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_9_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_9_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_9_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_9_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_9_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_9_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_9_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_9_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_10_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_10_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_10_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_10_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_10_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_10_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_10_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_10_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_10_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_10_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_10_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_10_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_10_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_10_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_10_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------

        #----------
        x1 = self.lrelu(self.RRDB_trunk_11_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_11_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_11_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_11_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_11_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_11_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_11_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_11_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_11_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_11_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_11_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_11_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_11_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_11_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_11_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_12_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_12_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_12_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_12_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_12_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_12_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_12_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_12_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_12_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_12_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_12_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_12_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_12_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_12_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_12_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_13_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_13_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_13_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_13_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_13_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_13_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_13_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_13_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_13_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_13_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_13_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_13_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_13_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_13_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_13_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------

        #----------
        x1 = self.lrelu(self.RRDB_trunk_14_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_14_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_14_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_14_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_14_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_14_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_14_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_14_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_14_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_14_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_14_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_14_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_14_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_14_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_14_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_15_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_15_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_15_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_15_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_15_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_15_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_15_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_15_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_15_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_15_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_15_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_15_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_15_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_15_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_15_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_16_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_16_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_16_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_16_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_16_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_16_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_16_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_16_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_16_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_16_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_16_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_16_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_16_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_16_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_16_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_17_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_17_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_17_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_17_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_17_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_17_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_17_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_17_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_17_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_17_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_17_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_17_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_17_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_17_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_17_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_18_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_18_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_18_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_18_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_18_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_18_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_18_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_18_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_18_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_18_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_18_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_18_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_18_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_18_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_18_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_19_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_19_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_19_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_19_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_19_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_19_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_19_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_19_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_19_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_19_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_19_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_19_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_19_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_19_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_19_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_20_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_20_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_20_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_20_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_20_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_20_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_20_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_20_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_20_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_20_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_20_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_20_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_20_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_20_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_20_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        #----------
        x1 = self.lrelu(self.RRDB_trunk_21_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_21_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_21_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_21_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_21_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_21_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_21_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_21_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_21_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_21_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_21_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_21_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_21_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_21_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_21_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------

        #----------
        x1 = self.lrelu(self.RRDB_trunk_22_RDB1_conv1(x_o))
        x2 = self.lrelu(self.RRDB_trunk_22_RDB1_conv2(torch.cat((x_o, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_22_RDB1_conv3(torch.cat((x_o, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_22_RDB1_conv4(torch.cat((x_o, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_22_RDB1_conv5(torch.cat((x_o, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x_o

        x1 = self.lrelu(self.RRDB_trunk_22_RDB2_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_22_RDB2_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_22_RDB2_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_22_RDB2_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_22_RDB2_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x

        x1 = self.lrelu(self.RRDB_trunk_22_RDB3_conv1(x))
        x2 = self.lrelu(self.RRDB_trunk_22_RDB3_conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.RRDB_trunk_22_RDB3_conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.RRDB_trunk_22_RDB3_conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.RRDB_trunk_22_RDB3_conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x = x5 * 0.2 + x
        x_o = x * 0.2 + x_o
        #----------


        trunk = self.trunk_conv(x_o)



        fea = fea + trunk
        return fea


        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
