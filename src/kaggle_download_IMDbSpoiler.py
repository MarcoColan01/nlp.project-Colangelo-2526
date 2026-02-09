'''
IMPORTANT: FOR AUTHENTICATION AND BE ABLE TO DOWNLOAD THE DATASET FROM KAGGLE, 
THIS .py MODULE REQUIRES THAT A kaggle.json FILE, CONTAINING user AND api_key FIELDS, IS PROVIDED ON src FOLDER

DATASET WILL BE SAVED ON data/raw DIRECTORY
'''


import os
import json 
import zipfile 
from kaggle.api.kaggle_api_extended import KaggleApi

try:
    with open('kaggle.json', 'r') as f:
        data = json.load(f)
        os.environ['KAGGLE_USERNAME'] = data['username']
        os.environ['KAGGLE_KEY'] = data['key']
except IOError:
    print(f"kaggle.json not found on src folder. Please provide your kaggle.json file")


api = KaggleApi()
api.authenticate()

dataset = "rmisra/imdb-spoiler-dataset"
print(f"Downloading {dataset}...")

api.dataset_download_files(dataset, path='../data/raw', unzip=True)