import os
import shutil
import json

from PIL import Image, ImageOps
import torch
from torchvision import transforms, datasets

HIERARCHY_DEPTH = 3
VALIDATE = 40

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

    for filename in os.listdir('./data'):
        with open(os.path.join(os.getcwd(), 'data', filename)) as json_data:
            data = json.load(json_data)
            hierarchy = data['hierarchy']
            if len(hierarchy) < HIERARCHY_DEPTH:
                # images should be duplicated with more specific taxonomic names anyway
                continue

            dirs = {}
            for directory in ['train', 'validate']:
                dirs[directory] = os.path.join(os.getcwd(), directory, hierarchy[HIERARCHY_DEPTH - 1])
                if not os.path.isdir(dirs[directory]): os.makedirs(dirs[directory])


            for images in data['samples']:
                for thumbnail in images['thumbs']:
                    thumbnail = thumbnail.split('/')[-1]
                    if not thumbnail: continue

                    # sample 1 in every VALIDATE images to use to assess model quality
                    if (train_count / validate_count) < VALIDATE:

                        shutil.copy(os.path.join(os.getcwd(), 'images', thumbnail),
                                    os.path.join(dirs['train'], thumbnail))
                        train_count += 1

                    else:
                        shutil.copy(os.path.join(os.getcwd(), 'images', thumbnail),
                                    os.path.join(dirs['validate'], thumbnail))
                        validate_count += 1


def create_dataloader(directory):
    """Use the structure of the folder created above, with generic ImageLoader, as per
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#afterword-torchvision
    """
    # These Normalize values are boilerplate everywhere, what do they signify?
    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    coccoliths = datasets.ImageFolder(root=directory,
                                      transform=data_transform)

    dataset_loader = torch.utils.data.DataLoader(coccoliths,
                                                 batch_size=4, shuffle=True,
                                                 num_workers=4)

    return dataset_loader


if __name__ == '__main__':
    prepare_imagefolder()
    create_dataloader('validate')




