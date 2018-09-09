# Google-Street-View-House-Number-Classifier
Goal: Classify the main digit that appears in each photo. There are 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10. 

# Information
The data set contains cropped images, corresponding to street numbers taken from images of houses. The dataset is taken from: http://ufldl.stanford.edu/housenumbers/

# Files
train.mat A Matlab data file containing the training data. The "X" trait contains the images (73257 images; each image is a 32x32 colour image, in matrix form). The "y" trait contains the labels (integer values 1...10).

test.mat A Matlab data file containing the testing data. The "X" trait contains the images, as in the training file; there are 26,032 test samples.

sample.csv A CSV file showing you what a sample submission should look like. 
