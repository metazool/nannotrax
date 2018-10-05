"""Dump out the layer descriptions for the currently used model configuration"""

from train_classifier import initialise_model

if __name__ == '__main__':
    model = initialise_model()
    sequence = []
    for module in model.modules():
        print(module)
