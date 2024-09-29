# CS 637 Term Project
"""SVHN Dataset
==============================
Preprocessing the 32 x 32 image datset
"""
import os
import tensorflow as tf
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import initializers
from sklearn.preprocessing import OneHotEncoder

# load the dataset
train = loadmat('train_32x32.mat')
test = loadmat('test_32x32.mat')

# extracting training/testing images and training/testing labels
train_imgs = train['X']
test_imgs = test['X']
train_labs = train['y']
test_labs = test['y']

# moving the 4th dimension as 1st
train_imgs = np.moveaxis(train_imgs, -1, 0)
test_imgs = np.moveaxis(test_imgs, -1, 0)

# selecting 10 random images from the training set and displaying them in a figure
index_imgs = list(np.random.randint(0, train_imgs.shape[0], [10]))
fig = plt.figure(figsize=(16, 16))
for i in range(10):
    fig.add_subplot(1, 10, i+1)
    plt.axis('off')
    plt.imshow(train_imgs[index_imgs[i],:,:,:])
    plt.title(train_labs[index_imgs[i]])
plt.show();

# converting training/test images to grayscale taking the mean
train_imgs_bw = np.mean(train_imgs, axis=3)[..., np.newaxis]
test_imgs_bw = np.mean(test_imgs, axis=3)[..., np.newaxis]

# normalizing images
train_imgs_bw = train_imgs_bw/255
test_imgs_bw = test_imgs_bw/255

# selecting 10 random images from the training set and displaying them in a figure
index_imgs = list(np.random.randint(0, train_imgs_bw.shape[0], [10]))
fig = plt.figure(figsize=(16, 16))
for i in range(10):
    fig.add_subplot(1, 10, i+1)
    plt.axis('off')
    plt.imshow(train_imgs_bw[index_imgs[i],:,:,:].squeeze(), cmap='gray')
    plt.title(train_labs[index_imgs[i]])
plt.show();

"""
MLP NEURAL NETWORK CLASSIFIER
"""
# first, we apply One Hot Encoder to our labels, to suit correctly our MLP with the final level of 10 neurons
encoder = OneHotEncoder().fit(train_labs)
train_labs_one_hot = encoder.transform(train_labs).toarray()
test_labs_one_hot = encoder.transform(test_labs).toarray()

test_labs_one_hot[0]

# building model
def build_mlp_model(shape):
#     Build an MLP model with a Flatten and four Dense layer
#     in - shape: the shape of the imput of our network
#     out - model: the model we just built
    model = Sequential()
    model.add(Flatten(input_shape=shape))
    model.add(Dense(512, activation='relu', kernel_initializer = initializers.RandomNormal(mean=0., stddev=0.1)))
    model.add(Dense(64, activation='relu', kernel_initializer = initializers.RandomNormal(mean=0., stddev=0.1)))
    model.add(Dense(32, activation='relu', kernel_initializer = initializers.RandomNormal(mean=0., stddev=0.1)))
    model.add(Dense(10, activation='softmax'))
    
    return model
                                 
model_mlp = build_mlp_model(train_imgs_bw[0].shape)

# printing out model summary
model_mlp.summary()

model_mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
checkpoint_dir = 'model_checkpoints_best'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# creating Tensorflow checkpoints object 
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
                             factor=0.5, 
                             patience=3,
                             verbose=1)
checkpoint_mlp_best_path = 'model_checkpoints_best/checkpoint_mlp.weights.h5'
checkpoint_best = ModelCheckpoint(filepath=checkpoint_mlp_best_path,
                                  save_weights_only=True,
                                  save_freq='epoch',
                                  monitor='val_accuracy',
                                  save_best_only=True,
                                  verbose=1)

# fitting the model for 30 epochs
history = model_mlp.fit(train_imgs_bw, 
                        train_labs_one_hot,
                        epochs=30, 
                        batch_size=64,
                        callbacks=[reduce_lr, checkpoint_best],
                        validation_data=(test_imgs_bw, test_labs_one_hot))

plt.plot(history.history['train_loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['loss','val_loss'], loc='upper right')
plt.show();


plt.plot(history.history['train_accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['accuracy','val_accuracy'], loc='upper left')
plt.show();

"""
CNN Neural Network Classifier

"""

# building model
def build_cnn_model():
#     Build a CNN with two convolutional + two max pooling layers, followed by two dense layers
#     out: model: the model we just built
    model = Sequential()
    model.add(Conv2D(16, 
                     kernel_size=(3,3), 
                     activation='relu', 
                     kernel_initializer = initializers.RandomNormal(mean=0., stddev=0.1), 
                     input_shape=(32, 32,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(16, 
                     kernel_size=(3,3), 
                     activation='relu',
                     kernel_initializer = initializers.RandomNormal(mean=0., stddev=0.1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    return model

model_cnn = build_cnn_model()

model_cnn.summary()

model_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# creating Tensorflow checkpoints object 
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
                              factor=0.5, 
                              patience=3,
                              verbose=1)
checkpoint_cnn_best_path = 'model_checkpoints_best/checkpoint_cnn.weights.h5'
checkpoint_best = ModelCheckpoint(filepath=checkpoint_cnn_best_path,
                                  save_weights_only=True,
                                  save_freq='epoch',
                                  monitor='val_accuracy',
                                  save_best_only=True,
                                  verbose=1)

# fitting the model for 10 epochs
history = model_cnn.fit(train_imgs_bw, 
                        train_labs_one_hot, 
                        epochs=10, 
                        batch_size=64,
                        callbacks=[reduce_lr, checkpoint_best],
                        validation_data=(test_imgs_bw, test_labs_one_hot))

plt.plot(history.history['train_loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['loss','val_loss'], loc='upper right')
plt.show();

plt.plot(history.history['train_accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['accuracy','val_accuracy'], loc='upper left')
plt.show();

"""
Model Predictions

"""
best_mlp = build_mlp_model(train_imgs_bw[0].shape)
best_mlp.load_weights(checkpoint_mlp_best_path)

best_cnn = build_cnn_model()
best_cnn.load_weights(checkpoint_cnn_best_path)

# selecting 5 random images from the test set and displaying them in a figure - MLP
index_imgs = list(np.random.randint(0, test_imgs.shape[0], [5]))
predictions_mlp = model_mlp.predict(test_imgs_bw[index_imgs])
predictions_cnn = model_cnn.predict(test_imgs_bw[index_imgs])

# showing MLP results
fig = plt.figure(figsize=(16, 16))
for i in range(5):
    fig.add_subplot(5, 2, 1+i*2)
    plt.axis('off')
    plt.imshow(test_imgs[index_imgs[i],:,:,:])
    plt.title(test_labs[index_imgs[i]]%10)
    fig.add_subplot(5, 2, (i+1)*2)
    pred = list(predictions_mlp[i])
    pred = pred[-1:] + pred[:-1]
    plt.bar(list(np.arange(10)), pred)
    plt.xticks(list(np.arange(10)))
plt.show();

# showing CNN results
fig = plt.figure(figsize=(16, 16))
for i in range(5):
    fig.add_subplot(5, 2, 1+i*2)
    plt.axis('off')
    plt.imshow(test_imgs[index_imgs[i],:,:,:])
    plt.title(test_labs[index_imgs[i]]%10)
    fig.add_subplot(5, 2, (i+1)*2)
    pred = list(predictions_cnn[i])
    pred = pred[-1:] + pred[:-1]
    plt.bar(list(np.arange(10)), pred)
    plt.xticks(list(np.arange(10)))
plt.show();


# Evaluation and Visualization Functions
def evaluate_model(model, test_images, test_labels):
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    cm = confusion_matrix(true_classes, predicted_classes)
    print(classification_report(true_classes, predicted_classes))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.show()


evaluate_model(model_mlp, test_imgs_bw, test_labs_one_hot)
evaluate_model(model_cnn, test_imgs_bw, test_labs_one_hot)
