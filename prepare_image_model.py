import os
import shutil
import json
import random
import logging

from PIL import Image, ImageOps
import torch
from torchvision import transforms, datasets

logging.basicConfig(level=logging.DEBUG)
HIERARCHY_DEPTH = 3
DATASETS = ['validate','testing','train','train','train']

BATCH_SIZE=20

def allocate_dataset():
    select = random.randrange(0, 4)
    return DATASETS[select]

def prepare_imagefolder():
    """Images should be categorized into subdirectories corresponding to labels.
    Finding out how narrowly we can classify the taxonomy will be trial and error
    We make a copy of them, sorted into directories for use with ImageLoader.
    The directory names serve as image labels.

    HIERARCHY_DEPTH indicates how narrow we want our taxonomic classification to be.
    If it's a larger value there will be more directories and more labels.
    (Default should be 2 really but our hierarchy data includes the mikrotax 'module')
    TODO remove module from the classification list if we ever re-scrape the data.
    """

    train_count = 0
    validate_count = 1

    class_images = {}

    for filename in os.listdir('./data'):
        with open(os.path.join(os.getcwd(), 'data', filename)) as json_data:
            data = json.load(json_data)

            hierarchy = data['hierarchy']

            if len(hierarchy) < HIERARCHY_DEPTH:
                # images should be duplicated with more specific taxonomic names anyway
                continue

            classname =  hierarchy[HIERARCHY_DEPTH - 1]

            for directory in ['train', 'validate', 'testing']:
                directory = os.path.join(os.getcwd(), directory, classname)
                if not os.path.isdir(directory): os.makedirs(directory)

            class_images[classname] = []

            for images in data['samples']:
                for thumbnail in images['thumbs']:
                    thumbnail = thumbnail.split('/')[-1]
                    if not thumbnail: continue
                    class_images[classname].append(thumbnail)

    for class_ in class_images:
        # Split between testing, training and validation
        for image in class_images[class_]:
            dataset = allocate_dataset()
            dest = os.path.join(os.getcwd(), dataset, class_, image)
            logging.debug(dest)
            shutil.copy(os.path.join(os.getcwd(), 'images', image), dest)


def create_imagefolder(directory):
    """Use the structure of the folder created above, with generic ImageLoader, as per
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#afterword-torchvision
    """

    # These Normalize values are boilerplate everywhere, what do they signify?
    # The 224 size is to coerce ResNet into working, but sources are all 120
    data_transform = transforms.Compose([
            #transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    coccoliths = datasets.ImageFolder(root=directory,
                                      transform=data_transform)

    return coccoliths


def create_dataloader(imagefolder):

    """Separate interface as we get the classnames from this interface"""
    dataset_loader = torch.utils.data.DataLoader(imagefolder,
                                                 batch_size=BATCH_SIZE, 
                                                 shuffle=True)#,
                                                 #drop_last=True)
                                                 #num_workers=4)

    return dataset_loader


if __name__ == '__main__':
    prepare_imagefolder()
