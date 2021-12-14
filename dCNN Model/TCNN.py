"""
Curated by TESTUBE.ai - Syed Farhan Hyder Abidi, Aravind R Krishnan, Andrea Clement, Sumukhi T
Mounting the drive
"""

from google.colab import drive
drive.mount("/content/gdrive")

"""
Defining the path for:
1) Training Set
2) Validation Set
3) Test Set
"""

train_path="#####"
validation_path="#####"
test_path = "#####"

"""
Importing all the DL libraries needed to build the model
"""

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
import keras
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2

"""
Initializing the CNN model as Sequential
""""

predictor = Sequential()

"""
Adding the following layers:
1) Convolution Layer - 64N
2) Max Pooling Layer
3) Dropout - 0.2
"""

predictor.add(Conv2D(64,(3,3), input_shape = (224,224,3), activation = 'relu', padding = 'same'))
predictor.add(MaxPooling2D(pool_size = (2,2)))
predictor.add(Dropout(0.2))

"""
Adding the following layers:
1) Convolution Layer - 512N
2) Max Pooling Layer
3) Dropout - 0.25
"""

predictor.add(Conv2D(512, (3, 3),activation = 'relu', padding='same', kernel_regularizer=l2(0.01)))
predictor.add(MaxPooling2D(pool_size = (2,2)))
predictor.add(Dropout(0.25))

"""
Adding the following layers:
1) Convolution Layer - 512N
2) Max Pooling Layer
3) Dropout - 0.25
"""

predictor.add(Conv2D(512, (3, 3),activation = 'relu', kernel_regularizer=l2(0.01), padding='same'))
predictor.add(MaxPooling2D(pool_size = (2,2)))
predictor.add(Dropout(0.25))

"""
Adding the following layers:
1) Convolution Layer - 512N
2) Max Pooling Layer
3) Dropout - 0.3
"""

predictor.add(Conv2D(512, (3, 3),activation = 'relu', kernel_regularizer=l2(0.01), padding='same'))
predictor.add(MaxPooling2D(pool_size = (2,2)))
predictor.add(Dropout(0.3))

"""
Adding the following layers:
1) Flattening Layer
2) Dense Layer - 64N
3) Output Layer - 1N
"""

predictor.add(Flatten())
predictor.add(Dense(64, activation ='relu', kernel_regularizer=l2(0.01)))
predictor.add(Dense(1, activation = 'sigmoid'))

predictor.summary()

"""
Pre-preocessing the dataset using keras.preprocessing
"""

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=30, class_mode='binary') 
validation_set = test_datagen.flow_from_directory(validation_path, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)
test_set = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)

"""
Compiling the model with adam as the optimizer
"""

predictor.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

"""
Training the model with the following params:
1) Epochs = 15
2) Batch Size = 32
3) Steps_per_epoch = Total dataset // Batch Size
""" 

history = predictor.fit_generator(training_set, 
                                  steps_per_epoch = 528//32, 
                                  epochs = 15, 
                                  validation_data = validation_set,  
                                  validation_steps = 67//32)

"""
Plotting the grapgh for:
1) Model Performance
2) Model Loss
"""

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Performance")
plt.xlabel("epoch")
plt.ylabel('Accuracy')
plt.legend(['acc', 'val_acc'])
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

"""
Importing Libraries for plotting CM
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, confusion_matrix

"""
Predicting the model perfomance over the test set
"""

from sklearn.metrics import classification_report

Y_pred = predictor.predict_generator(test_set)

"""
Cnverting the probabilities/predictions to binary (0/1)
"""

y_pred = np.where(Y_pred < 0.5, 0, 1)

"""
Computing the CM
"""

print('Confusion Matrix')
cm = confusion_matrix(test_set.classes, y_pred)
print(cm)

"""
Defining a func to visualize CM
"""

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
"""
Defining the labels for Cm
"""

cm_plot_labels = ['TB_Negative','TB_Positive']

"""
Plotting the CM
"""

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
