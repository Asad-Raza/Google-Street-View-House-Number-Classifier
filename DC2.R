# https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8
# https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
# https://www.kaggle.com/kentaroyoshioka47/cnn-with-batchnormalization-in-keras-94
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, regularizers
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout

# one hot encode outputs for y
y_train2 = to_categorical(y_train)
y_test2 = to_categorical(y_test)

num_pixels = 32*32
y_data2 = to_categorical(y_data)
num_classes = y_data2.shape[1]

# reshape and normalize inputs to 0 to 1
X_train2 = X_train.reshape(X_train.shape[3],32,32,3).astype('float32') /255
X_test2 = X_test.reshape(X_test.shape[3],32,32,3).astype('float32') /255
#x_data2 = x_data.reshape(x_data.shape[3],32,32,3).astype('float32') /255
x_data2 = np.transpose(x_data,(3,0,1,2)) /255


#def model():
# create model
model = Sequential()
#model.add(Conv2D(32, (3, 3), input_shape=(32, 32,3), activation='relu'))
model.add(Conv2D(32, (3, 3), input_shape=(32, 32,3), padding='same',activation='relu')) #32 to 64
model.add(Conv2D(32, (3, 3), input_shape=(32, 32,3),activation='relu')) #32 to 64
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), input_shape=(32, 32,3), padding='same',activation='relu')) #32 to 64
model.add(Conv2D(32, (3, 3), input_shape=(32, 32,3),activation='relu')) #32 to 64
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) 
model.add(Dense(512)) 

model.add(Activation('relu')) 
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
#model.fit(X_train2,y_train2,validation_data=(X_test2,y_test2),epochs=25,batch_size=128,callbacks=[early_stopping])
model.fit(x_data2, y_data2, validation_split=0.2,epochs=20,batch_size=200, callbacks=[early_stopping])