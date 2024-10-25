import os
import requests
import gzip
import hashlib
from PIL import Image
import numpy as np
from tqdm import tqdm

BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

os.makedirs('data', exist_ok=True)

DATA_DIR = 'data/MNIST/'

RESOURCES = [
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
]

def download_and_extract(file, md5):
    url = BASE_URL + file
    local_path = os.path.join(DATA_DIR, file)

    print(f"Downloading {file}...")
    response = requests.get(url)

    with open(local_path, 'wb') as f:
        f.write(response.content)

    computed_md5 = compute_md5(local_path)

    if computed_md5 == md5:
        print("MD5 checksum matches!")
    else:
        print("MD5 checksum does not match.")
        print(f"Computed MD5: {computed_md5}")
        print(f"Expected MD5: {md5}")

    with gzip.open(local_path, 'rb') as f_in:
        data = f_in.read()

    return data

def compute_md5(file_path, chunk_size=4096):
    "Compute the MD5 checksum of a file"
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()

def extract_images(data, metadata_bytes=16):
    """Extract and convert MNIST image data into numpy arrays."""
    # First 16 bytes are metadata
    magic_number, num_images, rows, cols = np.frombuffer(data[:metadata_bytes], dtype='>i4', count=4)
    print(f"Extracting {num_images} images of size {rows}x{cols}...")
    
    # The rest is the image data, each pixel is unsigned byte
    images = np.frombuffer(data[metadata_bytes:], dtype=np.uint8).reshape(num_images, rows, cols)
    
    return images

def extract_labels(data, metadata_bytes=8):
    # First 8 bytes are metadata
    magic_number, num_images = np.frombuffer(data[:metadata_bytes], dtype='>i4', count=2)
    print(f"Extracting {num_images} labels...")
    
    # The rest is the labels
    images = np.frombuffer(data[metadata_bytes:], dtype=np.uint8).reshape(num_images, )
    
    return images

def write_images(images, labels, split='train'):
    # create for folder for split
    split_path = os.path.join(DATA_DIR, split)
    os.makedirs(split_path, exist_ok=True)

    # create folder for classes
    for label in np.unique(labels):
        class_path = os.path.join(split_path, str(label))
        os.makedirs(class_path, exist_ok=True)

    for i, img in tqdm(enumerate(images), desc=f'Writing {split} images'):
        img_name = f"{i:{'0'}{6}}.png"

        img = Image.fromarray(img)
        img.save(os.path.join(split_path, str(labels[i]), img_name))


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)

    for file, md5 in RESOURCES:
        if 'images' in file:
            if 'train' in file:
                train_images = extract_images(download_and_extract(file, md5))
            else:
                test_images = extract_images(download_and_extract(file, md5))
        else:
            if 'train' in file:
                train_labels = extract_labels(download_and_extract(file, md5))
            else:
                test_labels = extract_labels(download_and_extract(file, md5))

        os.remove(os.path.join(DATA_DIR, file))

    write_images(train_images, train_labels, split='train')
    write_images(test_images, test_labels, split='test')


