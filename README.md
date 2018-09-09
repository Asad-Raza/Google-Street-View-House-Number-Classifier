# Google-Street-View-House-Number-Classifier
Goal: Classify the main digit that appears in each photo. There are 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10. 

# Information
The data set contains cropped images, corresponding to street numbers taken from images of houses. The dataset is taken from: http://ufldl.stanford.edu/housenumbers/

# Files
train.mat A Matlab data file containing the training data. The "X" trait contains the images (73257 images; each image is a 32x32 colour image, in matrix form). The "y" trait contains the labels (integer values 1...10).

test.mat A Matlab data file containing the testing data. The "X" trait contains the images, as in the training file; there are 26,032 test samples.

sample.csv A CSV file showing you what a sample submission should look like. 

# Method
I utilized a Convolutional Neural Network in order to classify each photo.
The SVHN data set provided was 32x32x3x73257 and I divided it into a test size of 35%. Using Python and the keras library, I was able to acheive a 91.791% accuracy according to Kaggle.

1) The first hidden layer is the convolutional layer (Conv2D). Based on the information above, it has 32 feature map and the size 32 x 32. There is also a rectifier activation function. This function expects image with the structure ouline of [pixels][widith][height]

2) Our pooling layer (MaxPooling2D) is configurated to be 2x2.

3) Our regularization layer, Dropout, is configured to randomly exclude 25% of neurons in the layer to reduce overfitting.

4) Our 2D matrix is converted into a vector using Flatten.

5) A fully connected layer with 512 neuron is next. This layer also has a rectifier activation function.

6) Last layter is an output layer with 10 neurons for the 10 number classes (0 - 9) and a softmax activation function to ouutput the probability prediction for each class. 

It was fitted with an epoch of 25 and a validation split of 20%.

https://www.kaggle.com/c/stat441-w2018-dc2/submissions?sortBy=date&group=all&page=1
