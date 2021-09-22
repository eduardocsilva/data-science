# Guitar Model Classification

## 1. Synopsis

This project served as an introduction to Tensorflow and its JavaScript implementation, allowing the trained model to be integrated in React.js front-end application, which can be consulted in the following [repository](https://github.com/eduardocsilva/guitar-classification-tensorflow) and [website](https://eduardocsilva.github.io/guitar-classification-tensorflow).

The obtained deep neural network model is capable of processing an image and classifying the guitar contained in it, with a relatively low 60% accuracy, which can be improved by collecting more data/images for each of the guitar models and finer tuning of the neural network's architecture and training process.

- **Keywords:** Machine Learning, Classification, Neural Network, Deep Learning, TensorFlow.

* **Based on the following [tutorial](https://www.tensorflow.org/tutorials/images/classification).**

---

**UPDATE (22/09/2022):** A PyTorch implementation was developed, in order to compare the differences between TensorFlow and PyTorch when applied to the same problem.

Curiously, the deep learning model obtained with PyTorch appears to perform better than TensorFlow's (96% accuracy vs 61% accuracy).

This might have happened due to the transformations / pre-processing applied to the images, since the neural networks have a similar structure.

---

## 2. Technologies

- Python

* Jupyter

- TensorFlow

* PyTorch

- Matplotlib

* etc.
