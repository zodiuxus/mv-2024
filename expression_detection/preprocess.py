import pandas as pd
import os, fnmatch
from pathlib import Path
import cv2
import numpy as np

def label_folders(inputFolderPath:str, outPath:str, outFile:str):
    """
    Used to append label to image based on the folder (class) it's in.
    Results in a .csv file with the columns 'filepath' and 'label' whose path is defined as /outPath/outFile.

    Automatically assigns a label to each folder containing images based on the alphabetical order they're in.
    Example:
        inputFolderPath
          |- train
              |- attribute1
              |- attribute2
              | ...
              |- attributeN
        which will result in a dictionary of {'attribute1': 0, 'attribute2': 1, ... 'attributeN': N-1}
    """
    if not os.path.isdir(inputFolderPath):
        raise FileNotFoundError(f"{inputFolderPath} could not be found.")

    if not os.path.isdir(outPath):
        os.makedirs(outPath)

    if not os.path.isfile(f"{outPath}/{outFile}"):
        dir_names_input = os.listdir(inputFolderPath)
        path_class_dict = {}

        for i in range(len(dir_names_input)):
            path_class_dict[dir_names_input[i]] = i
        df = pd.DataFrame(columns=["filename", "label"])

        for dirpath, dirs, filename in os.walk(f"{inputFolderPath}"):
            pathname = os.path.split(dirpath)[-1]
            if path_class_dict.get(pathname) is not None:
                current_class_temp_list = len(filename)*[path_class_dict.get(pathname)]
                df = pd.concat([pd.DataFrame(zip(filename, current_class_temp_list), columns=df.columns), df], ignore_index=True)

        df.to_csv(f"{outPath}/{outFile}", index=False)

    else: print(f"{outFile} already exists, skipping...")

def load_data(inFile:str, imagesFolder:str=None):
    """
        Loads training images from inFile.csv and returns a NumPy array of loaded images and their labels.

        Will recursively search for all files within the imagesFolder or the current working directory if not specified.
    """
    df = pd.read_csv(inFile)
    cwd = os.getcwd() if imagesFolder is None else imagesFolder

    loaded_images = []
    labels = []
    p = Path(cwd)
    print("Loading images...")
    for index, row in df.iterrows():
        image_path = list(p.rglob(row['filename']))
        image_loader = cv2.imread(image_path[0])
        if image_loader is not None:
            loaded_images.append(image_loader)
            labels.append(row['label'])
        else: print(f"Unable to load {image_path}.")
    
    return loaded_images, labels