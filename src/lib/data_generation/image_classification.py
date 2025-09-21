import requests
import zipfile
import os
import errno
from pathlib import Path

import numpy as np


def gen_img(start, target, fill=1, img_size=10):
    # Generates empty image
    img = np.zeros((img_size, img_size), dtype=float)

    start_row, start_col = None, None

    if start > 0:
        start_row = start
    else:
        start_col = np.abs(start)

    if target == 0:
        if start_row is None:
            img[:, start_col] = fill
        else:
            img[start_row, :] = fill
    else:
        if start_col == 0:
            start_col = 1

        if target == 1:
            if start_row is not None:
                up = (range(start_row, -1, -1), range(0, start_row + 1))
            else:
                up = (range(img_size - 1, start_col - 1, -1), range(start_col, img_size))
            img[up] = fill
        else:
            if start_row is not None:
                down = (range(start_row, img_size, 1), range(0, img_size - start_row))
            else:
                down = (range(0, img_size - 1 - start_col + 1), range(start_col, img_size))
            img[down] = fill

    return 255 * img.reshape(1, img_size, img_size)


def generate_dataset(img_size=10, n_images=100, binary=True, seed=17):
    np.random.seed(seed)

    starts = np.random.randint(-(img_size - 1), img_size, size=(n_images,))
    targets = np.random.randint(0, 3, size=(n_images,))

    images = np.array([gen_img(s, t, img_size=img_size) for s, t in zip(starts, targets)], dtype=np.uint8)

    if binary:
        targets = (targets > 0).astype(int)

    return images, targets


def download_rps(dir_path: Path):
    filenames = ["rps.zip", "rps-test-set.zip"]
    for filename in filenames:
        download_dir = dir_path / filename[:-4]

        if download_dir.exists() and download_dir.is_dir():
            continue

        download_dir.mkdir(parents=True, exist_ok=True)

        file_path = dir_path / filename
        # url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/{}'
        # Updated from TFDS URL at
        # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/rock_paper_scissors/rock_paper_scissors_dataset_builder.py
        url = "https://storage.googleapis.com/download.tensorflow.org/data/{}"
        r = requests.get(url.format(filename), allow_redirects=True)
        open(file_path, "wb").write(r.content)

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(dir_path)

        file_path.unlink()
