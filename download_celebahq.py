import os
import zipfile
import gdown

## Fetch data from Google Drive 
# Root directory for the dataset
data_root = 'data/celeba'
# Path to folder with the dataset
dataset_folder = f'{data_root}/img_align_celeba'
# URL for the CelebA dataset
url = 'https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view'
# Path to download the dataset to
download_path = f'{data_root}/celeba_hq.zip'

# Create required directories 
if not os.path.exists(data_root):
    os.makedirs(data_root)
    os.makedirs(dataset_folder)

# Download the dataset from google drive
gdown.download(url, download_path, quiet=False)

# Unzip the downloaded file 
with zipfile.ZipFile(download_path, 'r') as ziphandler:
    ziphandler.extractall(dataset_folder)