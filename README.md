# Group Normalization ResNet in Tensorflow with pre-trained weights on ImageNet

This repository contains codes of the **un-official** re-implementation of ResNet with [Group Normalization](https://arxiv.org/abs/1803.08494).

Group Normalization (GN) as a simple alternative to BN. GN's computation is independent of batch sizes, and its accuracy is stable in a wide range of batch sizes, which makes it outperform BN-based counterparts for object detection and segmentation in COCO.
##  ##
In order to accelerate further research based on Group Normalization, I would like to share the pre-trained weights on ImageNet for them, you can download from [Google Drive](https://drive.google.com/open?id=1MjXaQndjStRrfyKMNYS6_0xGfYYliZ2s). The pre-trained weights are converted from [official weights in Caffe2](https://github.com/facebookresearch/Detectron/tree/master/projects/GN) using [MMdnn](https://github.com/Microsoft/MMdnn) with other post-processing. And the outputs of all the network using the converted weights has almost the same outputs as original Caffe2 network (errors<1e-5). All rights related to the pre-trained weights belongs to the original author of [Group Normalization in Detectron](https://github.com/facebookresearch/Detectron/tree/master/projects/GN).

**This code and the pre-trained weights only can be used for research purposes.**

The canonical input image size for this ResNet is 224x224, each pixel value should in range [-128,128](BGR order), and the input preprocessing routine is quite simple, only normalization through mean channel subtraction was used. 

The codes was tested under Tensorflow 1.6, Python 3.5, Ubuntu 16.04. 

BTW, other scaffold need to be build for training from scratch. You can refer to [resnet/imagenet_main](https://github.com/tensorflow/models/blob/22ded0410d5bed85a88329e852cd20882593652b/official/resnet/imagenet_main.py#L189) for adding weight decay to the loss manually.
##  ##
Apache License 2.0
