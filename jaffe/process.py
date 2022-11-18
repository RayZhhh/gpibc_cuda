import argparse
import time

from PIL import Image
import os

import numpy as np

IH = 128
IW = 128

OUT_F = 'sad_train.txt'
SRC_P = 'sad_train/'

with open(OUT_F, 'w') as fout:
    for root, ds, fs in os.walk(SRC_P):
        for f in fs:
            image = Image.open(SRC_P + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((IH, IW))
            image_arr = np.array(image)  # 转化成numpy数组
            # image_arr = image_arr / 255
            image_arr = image_arr.reshape(-1)
            for i in image_arr:
                fout.write(str(i) + ' ')
            fout.write('\n')
