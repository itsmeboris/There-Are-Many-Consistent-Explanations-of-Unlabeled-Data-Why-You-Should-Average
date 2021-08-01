from glob import glob
import numpy as np
import shutil
import os
from PIL import Image, UnidentifiedImageError


def split_data():
    dirs = ['data_sets/cifar-10', 'data_sets/fruits', 'data_sets/Female and male eyes']
    nums_splits = [10, 3, 3]
    for dir, num_splits in zip(dirs, nums_splits):
        classes = [c.split('\\')[-1] for c in glob(f'{dir}/*')]

        for c in classes:
            images = np.asarray(glob(f'{dir}/{c}/*'))
            images_splits = np.array_split(images, num_splits)
            for i, images in enumerate(images_splits):
                output_dir = f'{dir}_{i}'
                os.makedirs(output_dir, exist_ok=True)
                output_dir = f'{output_dir}/{c}'
                os.makedirs(output_dir, exist_ok=True)
                for image in images:
                    shutil.copy(image, os.path.join(output_dir))


def remove_bad_imgs(m_dir="./Datasets"):
    for curr_dir, _, images in os.walk(m_dir):
        print(f"curr dir:  {curr_dir}")
        for image in images:
            ext = image.split(".")[-1]
            if ext != "png" and ext != "jpg":
                continue
            try:
                Image.open(os.path.join(curr_dir, image))
            except UnidentifiedImageError as e:
                msg = e.__str__()
                start_idx = msg.find("'")
                end_idx = msg.rfind("'")
                file = msg[start_idx+1: end_idx]
                os.remove(file)
                print(f"Removed file {file}")


if __name__ == '__main__':
    remove_bad_imgs()
