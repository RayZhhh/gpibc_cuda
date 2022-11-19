import argparse
import time

from PIL import Image
import os

import numpy as np

IH = 32
IW = 32

with open('0_test.txt', 'w') as fout:
    for root, ds, fs in os.walk('test_cat/'):
        for f in fs:
            image = Image.open('test_cat/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((IH, IW))
            image_arr = np.array(image)  # 转化成numpy数组
            # image_arr = image_arr / 255
            image_arr = image_arr.reshape(-1)
            for i in image_arr:
                fout.write(str(i) + ' ')
            fout.write('\n')

