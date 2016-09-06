# Human-Action-Recognition-with-Keras
Keras implementation of Human Action Recognition for the data set State Farm Distracted Driver Detection (Kaggle).

### DESCRIPTION:

This model uses 3 dense layers on the top of the convolutional layers of a pre-trained ConvNet (VGG-16) to classify driver actions into 10 classes. The dense layers are subject to an aggressive regularization by Eigenvalue Decay, the last 7 convolutional layers of the VGG-16 are fine tuned and data augmentation was applied on the training data (shear, zoom and rotation).

This model was ranked in the top 29% in the Kaggle private leaderboard after 80 training epochs. I believe the performance of this model could be much better if the model was trained during a larger number of epochs (since I adopted an aggressive regularization).

This script reuses pieces of code from the post "Building powerful image classification models using very little data", from blog.keras.io, and from https://www.kaggle.com/tnhabc/state-farm-distracted-driver-detection/keras-sample

### INSTALLATION AND USE:

1. Download the training data at https://www.kaggle.com/c/state-farm-distracted-driver-detection/data (keeping the name of the folders as "train" and "test");

2. Download the file "vgg16_weights.h5" containing the pre-trained weights of the VGG-16 at https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

3. Set the path for the file "vgg16_weights.h5" in line 32 of the code, variable "weights_path", and the number of epochs in line 57, I suggest more than 80 epoch, if you can;

4. This model uses the Eigenvalue Decay regularizer, if Keras is already installed, check if you have this regularizer, if you don't have it, update Keras to have Eigenvalue Decay: sudo pip install git+git://github.com/fchollet/keras.git --upgrade

5. Run “HumanActionRecognition.py” to train the deep model and create the submission file with the estimated classes for the test data. To run in GPU you can call the code like this: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high python HumanActionRecognition.py

