# These are my systematic notes

After experimenting with the network layout we drop back to a default VGG11 and try training on a very small sample (2 training and one validation image each for 2 classes) intending to do overfitting. 

Had to resample the images back up to 224*224

Set the BATCH_SIZE down to 2 so it would be less than the number of samples.

At this stage the validation loss/accuracy is never changing over 50 epochs.

We put the learn rate back up from the 0.0001 that Alex suggested to the default 0.001 and the momentum value (?) to the default 0.9. Epochs are running fast so let's try 500, it's only hardware, and prepare some food.






