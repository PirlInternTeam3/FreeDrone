# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import time
from PIL import Image

cwd = os.getcwd()
path=cwd+'/images/123.png'
files=glob.glob(path)

for f in files:
    with Image.open(f) as image:
        start = time.time()
        print("\n")

        pixels = np.array(image)
        crop = pixels[:, :]

        left = crop[:, :23]
        center = crop[:, 24:50]
        right = crop[:, 51:]

        left_reshape = left.reshape(-1)
        center_reshape = center.reshape(-1)
        right_reshape = right.reshape(-1)

        left_mean = left_reshape.mean()
        center_mean = center_reshape.mean()
        right_mean = right_reshape.mean()

        print("left_mean:", left_mean)
        print("center_mean:", center_mean)
        print("right_mean:", right_mean)

        path_dict = {"left":left_mean, "center":center_mean, "right":right_mean}
        max_value = max(left_mean,center_mean,right_mean)

        for k, v in path_dict.items():
            if v == max_value:
                print("image path:", f)
                print("\n")
                print("estimated path: fly to", k)

        print("\n")
        print("process time:", time.time() - start, "seconds per frame")
        print("\n")