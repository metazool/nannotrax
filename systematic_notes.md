# These are my systematic notes

After experimenting with the network layout we drop back to a default VGG11 and try training on a very small sample (2 training and one validation image each for 2 classes) intending to do overfitting. 

Had to resample the images back up to 224*224

Set the BATCH_SIZE down to 2 so it would be less than the number of samples.

At this stage the validation loss/accuracy is never changing over 50 epochs.

We put the learn rate back up from the 0.0001 that Alex suggested to the default 0.001 and the momentum value (?) to the default 0.9. Epochs are running fast so let's try 500, it's only hardware, and prepare some food.

We let it run for 500 epochs but the training loss fluctuates at a high value and validation loss is constant. The advice about training a very small sample set to 0 isnt clear but the received wisdom is that one needs 1000 samples per class to train VGG from scratch so next step before getting out Tensorboard is to try this on a pre-trained network.

After this stage we notice that a) the validation loss is never changing over iterations and the training loss is hovering around over up to a few hundred epochs and never showing a consistent downward trend b) the samples may be a poor fit as the two classes were selected from 27 (looking 2 levels down the navigation hierarchy for the taxonomy) and to the untrained eye they look very similar. So before moving on we will lower the hierarchy depth and have more distinct samples (currently 2 for training and 1 for validation)

The validation accuracy not changing is awkward as this is used to save the model so i think the current logic means we always end up saving the model from the first epoch. One run out of a few got 1.0 accuracy but this was likely a lucky guess, but that probably justifies poking around with Tensorboard. (We looked briefly into the Crayon setup but it looks overengineered for our purposes)

While preparing to add Tensorboard logging for a tiny set we realise that the batch_size is set to smaller than the size of the validation set (which was just one image for 2 training images). Refactored the preparation code a bit to select the tiny training set more randomly, not sure whether to set the batch size automatically from a ratio of the dataset size. And now this training run is looking A LOT healthier, training loss is not 0 but is hovering around 0.001. The validation loss dropped consistently for the first few runs, then rose to stick at one value with 0.75 accuracy. This is progress.








