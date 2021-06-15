Note: This works very well. I've not been not maintaining it for a while. I'll give you gals and guys some pointers.
- OpenCV Version
- Tensorflow Version (This is a major problem. This code was written for a really really old version. I don't even remeber it. I did it in a trance!)

Facenet C++ Classifier
====================
- This is C++ implementation of the Facenet classifier for the Facenet project by David Sandberg https://github.com/davidsandberg/facenet
- Part of the Tensor Flow code has been taken from https://github.com/davidsandberg/facenet/issues/357 and https://github.com/tensorflow/tensorflow/issues/8033
- Preprocessing of Images is done using code posted by knighthappy on https://github.com/davidsandberg/facenet/issues/357
- It has been compiled & run using OpenCV 3.3.0 and Tensor Flow 1.4.0 on Fedora 26 x86_64 and Ubuntu 16.04 x86_64
- It has been also adapted for OpenCV 4

Build
====================
    cd src && mkdir build && cd build
    cmake ..
    make

Usage
====================
- ./run <TRAIN|CLASSIFY>  <Path/To/TensorFlowModel> <Path/To/Image/Directory/Structure> <Path/To/Classifier/Model>
- Directory structure should be <class_id>/<image_files>
- Face images should be 160x160, otherwise they will be resized

Issues
====================
- More Testing Needed
- Results are good with more than one input image given to the pre-trained model.

Instructions to compile Tensor Flow C++ shared library
=====================
- https://www.tensorflow.org/install/install_sources to get libtensorflow_framework.so in /usr/lib/python2.7/site-packages/tensorflow/
- http://tuatini.me/building-tensorflow-as-a-standalone-project/
- https://github.com/FloopCZ/tensorflow_cc <- this one was used here

