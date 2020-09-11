---
published: true
---
MNIST ("Modified National Institute of Standards and Technology") is the de-facto “hello world” data set of computer vision. Since its release in 1999, this classic data set of handwritten images has served as the basis for bench marking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

In this competition, your goal is to correctly identify digits from a data set of tens of thousands of handwritten images. We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.

This Notebook contains three main parts:

-The data preparation

-The CNN modeling and evaluation

-The results prediction and analysis

### Data Preparation

The Google Drive is mounted on the Google Colab Notebook. The train & test data is then loaded using the well known pandas read_csv(). The train set 42000 rows and 785 columns( including labels) and the test set has 28000 rows and 784 columns. As known, the test set is not supposed to have the labels.

After importing the data sets, the label column from the train set is saved as another data frame, where as it is dropped from the original train set. This is merely done due to input requirements of the fit() function which takes the data & the labels separately during the model training phase.

## Distribution of Training data among Classes

The training data seems to be distributed equally among the classes ranging from 0 - 9.

## Checking for Null Data

We checked for null data, but did not find any. No further actions taken in this regard.

## Normalization

We perform a gray scale normalization to reduce the effect of illumination's differences. Moreover the CNN converge faster when pixel values ranges in [0, 1] rather than on [0, 255].

## Reshape

Train and test images has been stock into pandas dataframe as 1D vectors of 784 values. We reshape all data to (28x28x1) 3D matrices. Keras(with TensorFlow back end) requires an extra dimension in the end which correspond to channels. MNIST images are gray scaled so it use only one channel. Had it been colored or RGB images containing 3 channels each corresponding to Red, Green & Blue, we would have reshaped 784 vectors to (28x28x3) 3D matrices.

## Label Encoding

Labels are 10 digits numbers from 0 to 9. We need to encode these labels to one hot vectors.

## Splitting the Train Set

Next we split the train set into train and test in the ratio of 9:1. This 10% set will act as the validation set, to prevent our model from over fitting on the train set.

### CNN Modelling & Evaluation

The first is the convolution (Conv2D) layer. It is like a set of learnable filters. We chose to set 64 filters for the two firsts Conv2D layers and 64 filters for the lat one. Each filter transforms a part of the image as defined by the kernel size(3x3) using the kernel filter. The kernel filter matrix is applied on the whole image. Filters are used to transform the image.

The CNN can isolate features that are useful everywhere from these transformed images (feature maps).

The second important layer in CNN is the pooling (MaxPool2D) layer. This layer simply acts as a down sampling filter. It looks at the 2 neighboring pixels and picks the maximal value. These are used to reduce computational cost, and to some extent also reduces over fitting. We chose the pooling size as (2x2).

Combining convolution and pooling layers, CNN are able to combine local features and learn more global features of the image.

Dropout is a regularization method, where a proportion of nodes in the layer are randomly ignored (setting their weights to zero) for each training sample. This drops randomly a proportion of the network and forces the network to learn features in a distributed way. This technique also improves generalization and reduces the over fitting.

'relu' is the rectified linear activation function Mathematically it can be defined as f(x) = max(0,x). The activation function is used to add non linearity to the network.

The Flatten layer is use to convert the final feature maps into a one single 1D vector. This flattening step is needed so that you can make use of fully connected layers after some convolution/maxpool layers. It combines all the found local features of the previous convolution layers.

In the end, two fully-connected (Dense) layers are used which are just artificial an neural networks (ANN). In the last dense layer, 'softmax' activation is used which is used to output distribution of probability of each class.

Here is the Sequential model created:

```
model= tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', input_shape = (28,28,1)),
     tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'),
     tf.keras.layers.MaxPooling2D(2, 2),
     tf.keras.layers.Dropout(.25),
     tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'),
     tf.keras.layers.MaxPooling2D(2,2),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dense(10, activation='softmax')]
)
```
## Loss Function & Optimizer

We choose the loss function as 'categorical cross entropy' as the number of categorical classifications is more than 2. The loss function is the error rate between the observed labels and the predicted ones.

We used the Adam Optimizer for this particular problem.

## Model Summary

```
model.summary()

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 64)        640       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 64)        36928     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 12, 12, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 10, 10, 64)        36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               204928    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 280,714
Trainable params: 280,714
Non-trainable params: 0
_________________________________________________________________
```
## Train the model

The fit function is responsible for training the model. It takes as input the train data as x and the labels as y. In addition to this, we use a validation set as discussed earlier. In addition to this choose batch size as 100, epochs = 24 and verbose = 2. We trained the model using the GPU available on Google Colab.
```
model.fit(X_train, Y_train,validation_data=(X_test, Y_test), batch_size=100, epochs=24, verbose=2)
```
### Result Prediction & Analysis

After training the model, at the 24th epoch, the training accuracy is 99.8% and the validation accuracy is 98.9%. This shows that our model did not over fit much. Accuracy can be further improved if Data Augmentation is performed.

## Confusion Matrix

We plot the confusion matrix and found out some errors made during classification task performed on the validation set. It seems that our CNN has some little troubles with the 4 digits, hey are mis-classified as 9. Sometime it is very difficult to catch the difference between 4 and 9 when curves are smooth.
```
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

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_test,axis = 1)   
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 
```
Please access the [GitHub](https://github.com/saptarshidatta96/DigitRecognizerCNN) Repository to view the code. It's self Explanatory along with the description given here.

© 2020 Saptarshi Datta. All Rights Reserved.