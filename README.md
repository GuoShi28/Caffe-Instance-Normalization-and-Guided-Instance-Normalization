# (Caffe) Instance Normalization and Guided Instance Normalization

I implement Instance Normalization and Guided Instance Normalization of "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization " in caffe. 

Paper: https://arxiv.org/abs/1703.06868

## Instance Normalization

Instance Normalization is a recently widely used in CNN to avoid the drawback of batch normalization which brings in-batch correlations. The details can be found in Sec. 3.2 of the paper.

The implement are: "instance_norm_layer.cpp", "instance_norm_layer.cu" and "instance_norm_layer.hpp". 

To note that, since the implement of Batch normalization in caffe are separated into 2 layer, i.e., "BatchNorm" and "Scale".

Scale layer is also needed for our implement of instance_norm_layer.

## Guided Instance Normalization

Guided Instance Normalization can be used in style transfer.

More details can be read in Sec. 3.3 of the paper.

The \gamma and \beta are learned from style images instead of "Scale" layer.

We write 2 other layers , i.e., "GuidedInstanceNorm" and "AdaScale".

GuidedInstanceNorm is used to calculate the mu and var of style images as \gamma and \beta.

And AdaScale is used for Equ. 7 in the paper.
