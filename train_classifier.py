"""Frankly cobbled together from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html"""

import time
import os
import copy
import argparse
import os
import logging

import torch
import torch.nn as nn
from VGG import vgg_custom
from SimpleCNN import simple_cnn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from prepare_image_model import create_dataloader, create_imagefolder

EPOCHS=500
LEARN_RATE=0.001
MOMENTUM=0.9
INPUT_SIZE=(120, 120)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_data(directory=None):
    path = [os.getcwd()]

    if directory:
        path.append(directory)

    dirs = {
        'train': os.path.join(*path, 'train'),
        'validate': os.path.join(*path, 'validate')}

    images = {x: create_imagefolder(dirs[x]) for x in ['train', 'validate']}
    data = {x: create_dataloader(images[x]) for x in ['train', 'validate']}

    return (images, data)

def train_model(model, images, datsets, criterion, optimizer, scheduler, num_epochs=15):

    dataset_sizes = {x: len(images[x]) for x in ['train', 'validate']}

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validate']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.

            for inputs, labels in datasets[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()

 # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validate' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def initialise_model(images):
    """Create an empty model - TODO specify num_classes from datset
    models.vgg11 =
    vgg_custom - VGG with custom AvgPool to avoid 224x224 limiti
    simple_cnn - simplest thing that might possibly work"""
    classes = len(images['train'].classes)
    logging.info(f'training model with {classes} classes')
    model = models.vgg11(num_classes=classes)

    return model


def build_model(images, datasets, epochs=EPOCHS):
    """Run the training regime on the model and save its best effort"""
    model_ft = initialise_model(images)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, images, datasets, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=epochs)

    torch.save(model_ft, os.path.join(os.getcwd(),'model'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train a model based on contents of a directory")
    parser.add_argument(
        '--directory',
        help="Path of a directory on this host")

    args = parser.parse_args()
    images, datasets = prepare_data(directory=args.directory)
    build_model(images, datasets)

