"""Frankly cobbled together from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html"""

import time
import os
import copy
import argparse
import os
import logging

import torch
import torch.nn as nn
#from VGG import vgg_custom
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

from prepare_image_model import create_dataloader, create_imagefolder
from logger import Logger

logging.basicConfig(level=logging.DEBUG)

EPOCHS=100
LEARN_RATE=0.001
MOMENTUM=0.9
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOGGER=Logger('./logs')

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

def train_model(model, images, datsets, criterion, optimizer, scheduler, num_epochs=15, log=False):

    dataset_sizes = {x: len(images[x]) for x in ['train', 'validate']}

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    step_count = 0

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

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Sttempting to add Tensorboard logging her
                optimizer.step()
                if log:
                    step_count += 1
                    tensorboard_logging(step_count, model, inputs, running_loss, running_corrects.double())

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


def tensorboard_logging(step=None, model=None, images=None, loss=None, accuracy=None):
    """Write logs for this step so we can see stae via Tensorboard"""
    # 1. Log scalar values (scalar summary)
    info = { 'loss': loss, 'accuracy': accuracy }

    for tag, value in info.items():
        LOGGER.scalar_summary(tag, value, step+1)

    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        LOGGER.histo_summary(tag, value.data.cpu().numpy(), step+1)
        LOGGER.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step+1)

    # 3. Log training images (image summary)
    info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }

    for tag, images in info.items():
        LOGGER.image_summary(tag, images, step+1)


def initialise_model(images):
    """Create an empty model - TODO specify num_classes from datset
    models.vgg11 =
    vgg_custom - VGG with custom AvgPool to avoid 224x224 limiti
    simple_cnn - simplest thing that might possibly work"""
    classes = len(images['train'].classes)

    logging.info(f'training model with {classes} classes')

    # Try resetting the last fully connectted layer on a pre-trained VGG11
    model = models.vgg11(pretrained=True) #num_classes=classes, pretrained=True)

    # VGG specific logic, last layer in  self.classifier
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, classes)

    model = model.to(DEVICE)

    return model


def build_model(images, datasets, epochs=EPOCHS, log=False):
    """Run the training regime on the model and save its best effort"""
    model_ft = initialise_model(images)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, images, datasets, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epochs, log=log)

    torch.save(model_ft, os.path.join(os.getcwd(),'model'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train a model based on contents of a directory")
    parser.add_argument(
        '--directory',
        help="Path of a directory on this host")
    parser.add_argument(
        '--log',
        action='store_true')

    args = parser.parse_args()

    if args.log:
        print('logging on')
    images, datasets = prepare_data(directory=args.directory)
    build_model(images, datasets, log=args.log)

