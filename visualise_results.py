# Mostly the visualisation parts of the transfer learning tutorial
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torchvision

from prepare_image_model import create_dataloader, create_imagefolder

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def validation_data(imagefolder):
    return create_dataloader(imagefolder)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualise_model(model, num_images=12, directory='validate'):
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    images = create_imagefolder(directory)
    class_names = images.classes

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_data(images)):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            print(images_so_far)
            if images_so_far >= num_images-1:
                plt.tight_layout()
                plt.savefig('test.png')
                return

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//4, 4, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {} \n actual: {}'.format(class_names[preds[j]], class_names[labels[j]]), fontsize=8)
                imshow(inputs.cpu().data[j])


def load_model(model_name):
    return torch.load(model_name)


if __name__ == '__main__':
    visualise_model(load_model('model'))
