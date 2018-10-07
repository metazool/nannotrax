# Introduction

[Nannotax](http://www.mikrotax.org/Nannotax3/index.html) is a *authoritative guide to the biodiversity and taxonomy of coccolithophores* hosted on the Mikrotax platform.

This repository is not affiliated with Mikrotax in any way, it contains an unlicensed and crude screen-scraper that extracts a minimal amount of structured data from that project's web pages.

This was created with the intention of producing a labelled training dataset of low-resolution coccolith images for use with [Torchvision](https://pytorch.org/docs/stable/torchvision/datasets.html)

The repository includes experiments in training pytorch neural nets to recognise different classes of coccoliths. There's a training routine expanded from the transfer learning tutorial and also a misguided attempt to tweak the default VGG to add an adaptive average pool in place of some of the max pools so the images can be of arbitrary size (the source is a beautifully prepared collection of 120x120 thumbnails of scanning electron microscope images).

On the excellent advice of Simon Cozens there is now also a rambling set of systematic notes describing what changes are being made in response to what behaviours of the neural net / observations about the training data.




