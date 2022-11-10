# Convolutional Neural Network and Computer Vision with TensorFlow
This repository contains introductory topics for Convolutional Neural Network and Computer Vision with TensorFlow

# What is CNN ?

CNN, short for, Convolutional Neural Network, is a special kind of neural network which is used for computer vision (detecting patterns in visual data).

Since we are surrounded by cameras, capturing images and videos of every moment, we have huge of amount of data in image (or video format), and to detect patterns and apply algorithms to make a productive use of those type of data, we need specific kind of model architectures to fulfil our need, and that is what Convolutional Neural Network architectures comes into action.

For example, you might want to:
* Classify whether a picture of road contains car 🚗 or bicycle 🚲 or truck 🚛 (we're going to do this)
* Detect whether or not an object appears in an image (e.g did a specific car pass through a security camera?)

In this repo, we're going to follow the TensorFlow modelling workflow we've been following so far whilst learning about how to build and use CNNs.

# (typical)* Architecture of a CNN


| **Hyperparameter/Layer type** | **What does it do?** | **Typical Values** | 
| --- | --- | --- |
| Input image(s) | Target images you'd like to discover patterns in | Whatever you can take a photo(or video) of |
| Input layer | Takes in target images and preprocess them for further layers | `input_shape = [batch_size, image_height, image_width, color_channels]` |
| Convolution layer | Extracts/learns the most important features from target images | Multiple, can create with [tf.keras.layers.ConvXD](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) (X can be multiple values) |
| Hidden activation | Adds non-linearity to learned features (non-straight lines) | Usually [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu) |
| Pooling layer | Reduces the dimensionality of learned image features | Average [tf.keras.layers.AvgPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D) or Max [tf.keras.layers.MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D) |
| Fully Connected layer | Further refines learned features from convolution layers | [tf.keras.layers.Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) |
| Output Layer | Takes learned features and outputs them in shape of target labels | `output_shape = [number_of_classes]` (e.g. 3 for pizza, steak or sushi) |
| Output Activation | Adds non-linearities to output layer | [tf.keras.activations.sigmoid](https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid) (binary classification) or [tf.keras.activations.softmax](https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax) |

How they stack together
![image](https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-simple-convnet.png)

# What we're going to cover

Specifically, we're going to go through the follow with TensorFlow:

* Getting a dataset to work with
* Architecture of a convolutional neural network
* A quick end-to-end example (what we're working towards)
* Steps in modelling for binary image classification with CNNs
  - Becoming one with the data
  - Preparing data for modelling
  - Creating a CNN model (starting with a baseline)
  - Fitting a model (getting it to find patterns in our data)
  - Evaluating a model
  - Improving a model
  - Making a prediction with a trained model
* Steps in modelling for multi-class image classification with CNNs
  - Same as above (but this time with a different dataset)
  
# Notebook/Practicals

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaketMunda/cnn-computer-vision-introduction/blob/master/convolutional_neural_network_with_tensorflow.ipynb)

# Exercises

# Extra-Curriculam

# Resources
* [TensorFlow for Deep Learning by Daniel Bourke](https://dev.mrdbourke.com/tensorflow-deep-learning/03_convolutional_neural_networks_in_tensorflow/)
* [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
