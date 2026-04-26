import os, shutil
import cv2
import numpy as np 
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare results')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    filename = args.input

    data = np.load(filename)["arr_0"]

    dpath = args.outdir 
    os.makedirs(dpath, exist_ok=True)

    for i, img in enumerate(data):
        sf = os.path.join(dpath, '{}.png'.format(i))
        # img = img[:, :, 0]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sf, img)

    print(data.shape)
    print(args.input, " -> ", args.outdir, "Done!")
