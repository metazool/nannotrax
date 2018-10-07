"""Dump out the layer descriptions for the currently used model configuration"""

from train_classifier import initialise_model
from prepare_image_model import create_imagefolder

if __name__ == '__main__':
    model = initialise_model({'train':create_imagefolder('train')})
    sequence = []
    #for module in model.modules():
    #    print(module)

    print(model.classifier[-1])
