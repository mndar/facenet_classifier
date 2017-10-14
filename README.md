Facenet C++ Classifier
====================
- This is C++ implementation of the Facenet classifier for the Facenet project by David Sandberg https://github.com/davidsandberg/facenet
- Part of the Tensor Flow code has been taken from https://github.com/davidsandberg/facenet/issues/357 and https://github.com/tensorflow/tensorflow/issues/8033
- It has been compiled & run using OpenCV 3.3.0 and Tensor Flow 1.3.0 on Fedora 26 x86_64

Usage
====================
- ./facenet_classify <TRAIN|CLASSIFY>  <Path/To/20170512-110547.pb> <Path/To/Image/Directory/Structure> <Path/To/SVM/Model> <Path/To/SVM/Labels>
- Directory structure should be <class_name>/<image_files>
- Face Images Should be 160x160

Issues
====================
- Little testing done
- I'm using the SVM implementaion in OpenCV 3.3.0 So, there is no prediction value I can extract unless it's a binary classification. I'm thinking of creating a branch that using KNN.

Instructions to compile Tensor Flow C++ shared library
=====================
- http://tuatini.me/building-tensorflow-as-a-standalone-project/
