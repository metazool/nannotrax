import os
import shutil
import json
import random
import argparse
import logging

from PIL import Image, ImageOps
import torch
from torchvision import transforms, datasets

logging.basicConfig(level=logging.DEBUG)

# Consider adpting this to subset bigger categories and merge smaller ones
HIERARCHY_DEPTH = 3 
DATASETS = ['validate','train','train','train','train']
DATA_DIR = os.path.join(os.getcwd(), 'data')
IMAGE_DIR = os.path.join(os.getcwd(), 'images')
TRAIN_DIR = os.getcwd()
BATCH_SIZE=2


def allocate_dataset():
    select = random.randrange(0, 4)
    return DATASETS[select]


def prepare_imagefolder(add_fuzz=0, limit_classes=0, limit_samples=0):
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

    for filename in os.listdir(DATA_DIR):
        with open(os.path.join(DATA_DIR, filename)) as json_data:
            data = json.load(json_data)

            hierarchy = data['hierarchy']

            if len(hierarchy) < HIERARCHY_DEPTH:
                # images should be duplicated with more specific taxonomic names anyway
                continue

            classname =  hierarchy[HIERARCHY_DEPTH - 1]

            if classname not in class_images:
                class_images[classname] = []

            for images in data['samples']:
                for thumbnail in images['thumbs']:
                    thumbnail = thumbnail.split('/')[-1]
                    if not thumbnail: continue
                    class_images[classname].append(thumbnail)

    classes = 0

    for class_ in class_images:
        samples = 0
        if limit_classes and classes >= limit_classes:
            break

        for directory in ['train', 'validate']:#, 'testing']:
            directory = os.path.join(TRAIN_DIR, directory, class_)
            if not os.path.isdir(directory): os.makedirs(directory)

        logging.debug(f'{class_}: {len(class_images[class_])}')
        # Split between testing, training and validation

        for image in class_images[class_]:
            if limit_samples and samples >= limit_samples:
                break

            if add_fuzz:
                # try Vyron's suggestion of altered copies to bulk out dataset
                variants = fuzzed_images(images, add_fuzz)
                for v in variants: copy_image(v, class_)

            copy_image(image, class_)
            samples += 1

        classes += 1


def copy_image(filename, label_dir):
    """Copy source to labelled directory, randomly allocated to validation or training"""
    dataset = allocate_dataset()
    dest = os.path.join(TRAIN_DIR, dataset, label_dir, filename)
    logging.debug(dest)
    shutil.copy(os.path.join(IMAGE_DIR, filename), dest)


def fuzzed_image(filename, num_variants):
    """Randomly flip or rotate up to num_variants copies of the image"""
    return []


def create_imagefolder(directory):
    """Use the structure of the folder created above, with generic ImageLoader, as per
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#afterword-torchvision
    """

    # These Normalize values are boilerplate everywhere, what do they signify?
    # The 224 size is to coerce torchvision models into working, but sources are all 120
    data_transform = transforms.Compose([
            transforms.Resize(224),
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
                                                 shuffle=True,
                                                 num_workers=True)#,
                                                 #drop_last=True)
                                                 #num_workers=4)

    return dataset_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepare sample images in the ImageFolder per-class layout")
    parser.add_argument(
        '--data',
        help="Optional path of a directory on this host with JSON source data")
    parser.add_argument(
        '--images',
        help="Optional path of a directory on this host with image files")
    parser.add_argument(
        '--train',
        help="Optional path of a directory to lay out training data in")
    parser.add_argument(
        '--class_limit',
        type=int,
        help="limit to this number of classes")
    parser.add_argument(
        '--sample_limit',
        type=int,
        help="limit to this number of samples per class")

    args = parser.parse_args()
    if args.data:
        DATA_DIR=os.path.join(os.getcwd(), args.data)
    if args.images:
        IMAGE_DIR=os.path.join(os.getcwd(), args.images, 'images')
    if args.train:
        TRAIN_DIR=os.path.join(os.getcwd(), args.train)

    prepare_imagefolder(limit_classes=args.class_limit,
                        limit_samples=args.sample_limit)
