# Segmentation from Superpixels

Building a segmentation network using SLIC Superpixels as input to represent a classifier for superpixels. The network should work such that when given an image, computes superpixels and classifies each superpixel as one of the 9 classes of MSRC v1.


## Steps to implement this approach
There are basically three steps in the training stage.
1. Dilate and save each superpixel region from SLIC output into image of size 224X224, along with the ground truth segments label.
2. Use a pre-trained VVG-16 network by assing more convolution layers to extract the deep features from each Superpixel patch image with the last layer being Fully connected layers.
3. To define the segmentation loss as multi-class classification loss and train a convolutional neural network based classifier.
4. Combining the classifier's predicted labels to form the whole input image's Superpixel segmentation results during inference.

## Creating a Superpixel Dataset
As part of this task the smallest rectangle enclosing the superpixel is found and subsequently dilated by 3 pixels. This patch is converted into ‘.npy’ format and saved. Similarly, the same region is obtained from the segmentation ground truth image to obtain the corresponding class label.

<p float="left">
<img src="https://github.com/jayesh68/SUPERPIXL-SEGMENTATION-CLASSIFICATION/blob/main/Data/horse_segmentation.png" width="250" height="300"/>
<img src="https://github.com/jayesh68/SUPERPIXL-SEGMENTATION-CLASSIFICATION/blob/main/Data/horse_segmentation1.png" width="250" height="300"/>
<img src="https://github.com/jayesh68/SUPERPIXL-SEGMENTATION-CLASSIFICATION/blob/main/Data/horse_segmentation2.png" width="250" height="300"/>
</p>

## Deep network
A pre-trained VGG network is used, and the last few layers are replaced with a fully connected layer. A fully connected layer with 1024 units has been added.

## Results
The data is split randomly with 80% in the training and the remaining 20% in the test set.
<p float="left">
<img src="https://github.com/jayesh68/SUPERPIXL-SEGMENTATION-CLASSIFICATION/blob/main/Data/supsegout1.png" width="250" height="400"/>
<img src="https://github.com/jayesh68/SUPERPIXL-SEGMENTATION-CLASSIFICATION/blob/main/Data/supsegout2.png" width="250" height="400"/>
<img src="https://github.com/jayesh68/SUPERPIXL-SEGMENTATION-CLASSIFICATION/blob/main/Data/supsegout3.png" width="250" height="400"/>
</p>
