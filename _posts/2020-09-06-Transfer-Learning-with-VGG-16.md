---
layout: post
title: Transfer Learning with VGG-16
published: true
---

Training large models with huge number of trainable parameters is often time consuming. Sometimes, model training may go on for a couple of days or even more. In that case, if after such long training period, if the moel does not perform suitably, we again need to modify and re-train the model for such longer duration. This phenomenon results in wastege of time and efforts.

We do have a technique called Transfer Learning for quicker model training and hence faster experimentation. Various pretrained models like VGG-16, VGG-19, ResNet etc. are available with the weights of their corrspondinhg layers. We need to use the weights of the earlier layers and onlu need to train the outer layers with the output layers. This is because, most of the models determine simple features at their initial/earlier layers which are same for most of the tasks. Hence, we do not need to train them again.

We have prepared a notebook to answer a problem where we try to distinguish between a cat and a dog. We first implement our own Sequential model, train it and note down the accuracy. Then, we implement a VGG-16 Model with all pretrained parameters. We remove the last/output layer which initially has a softmax activation of 1000 outputs and replace it by a sigmoid activation for predicting a cat or a dog. We will train only the last layer and keep the parameters of the previous layers as it is.

Logically, I don't see any real life benefit by solving the above problem except understanding the benefits of Transfer Learning. That is what, I specifically want to show with this exercise.

**Our own sequential model**

The first is the convolution (Conv2D) layer. It is like a set of learnable filters. We chose to set 64 filters for the two firsts Conv2D layers and 64 filters for the lat one. Each filter transforms a part of the image as defined by the kernel size(3x3) using the kernel filter. The kernel filter matrix is applied on the whole image. Filters are used to transform the image.

The CNN can isolate features that are useful everywhere from these transformed images (feature maps).

The second important layer in CNN is the pooling (MaxPool2D) layer. This layer simply acts as a down sampling filter. It looks at the 2 neighboring pixels and picks the maximal value. These are used to reduce computational cost, and to some extent also reduces over fitting. We chose the pooling size as (2x2).

Combining convolution and pooling layers, CNN are able to combine local features and learn more global features of the image.

Dropout is a regularization method, where a proportion of nodes in the layer are randomly ignored (setting their weights to zero) for each training sample. This drops randomly a proportion of the network and forces the network to learn features in a distributed way. This technique also improves generalization and reduces the over fitting.

'relu' is the rectified linear activation function. Mathematically it can be defined as 
> f(x) = max(0,x)

The activation function is used to add non linearity to the network.

The Flatten layer is use to convert the final feature maps into a one single 1D vector. This flattening step is needed so that you can make use of fully connected layers after some convolution/maxpool layers. It combines all the found local features of the previous convolution layers.

In the end, two fully-connected (Dense) layers are used which are just artificial an neural networks (ANN). In the last dense layer, 'sigmoid' activation is used to predict whether the input image belongs to a cat or a dog.

```python
model= tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', input_shape = (224,224,3)),
     tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'),
     tf.keras.layers.MaxPooling2D(2, 2),
     tf.keras.layers.Dropout(.25),
     tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'),
     tf.keras.layers.MaxPooling2D(2,2),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dense(1, activation='sigmoid')]
```
The above model 23,963,777 trainable parameters and takes around 7 minutes to train per epoch.

```
model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 222, 222, 64)      1792      
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 220, 220, 64)      36928     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 110, 110, 64)      0         
_________________________________________________________________
dropout (Dropout)            (None, 110, 110, 64)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 108, 108, 64)      36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 54, 54, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 186624)            0         
_________________________________________________________________
dense (Dense)                (None, 128)               23888000  
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 129       
=================================================================
Total params: 23,963,777
Trainable params: 23,963,777
Non-trainable params: 0
_________________________________________________________________
```
Since the model takes around 7 minutes to train per epoch, we trained the model for only two epochs.

```
model.fit(x=train_batches,
          steps_per_epoch=len(train_batches),
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          epochs=2,
          verbose=2
)
Epoch 1/2
182/182 - 344s - loss: 0.6734 - accuracy: 0.5904 - val_loss: 0.6204 - val_accuracy: 0.6984
Epoch 2/2
182/182 - 332s - loss: 0.5923 - accuracy: 0.6782 - val_loss: 0.5587 - val_accuracy: 0.7304

<tensorflow.python.keras.callbacks.History at 0x7fa93117e550>
```
We find that both the train & validation accuracy is not that great. It's only around 69% to start with. You may train the model with longer epochs if time is not a constraint. Now, let's implment a VGG-16 pre-trained Model.

**VGG-16 pre-trained Model**

At first, we import the model. We need to have internet connection to import the model.
```
vgg16_model = tf.keras.applications.vgg16.VGG16()
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5
553467904/553467096 [==============================] - 5s 0us/step
```
Now, let's summrise the imported version of the VGG-16 Model.
````
vgg16_model.summary()
Model: "vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000   
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
````
We can see that teh VGG-16 modeel has 138,357,544 trainable parameters, but as said above, we are not going to train all of them. Instead, we are going to remove the last dense layer used for predictions, set rest of the layers as non-trainable and then add a sigmoid activation dense prediction layer with output shape (None, 1).
```
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
    
for layer in model.layers:
    layer.trainable = False
    
model.add(Dense(units=1, activation='sigmoid'))
```
Next, we print the summary of the model.
```
model.summary()

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 4097      
=================================================================
Total params: 134,264,641
Trainable params: 4,097
Non-trainable params: 134,260,544
```
We observe that, only 4097 parameters are trainable. This will considerably improve the training speed along with the accuracy of the model. We will see this next.
```
model.fit(x=train_batches1,
          steps_per_epoch=len(train_batches1),
          validation_data=valid_batches1,
          validation_steps=len(valid_batches1),
          epochs=3,
          verbose=2)
Epoch 1/3
182/182 - 114s - loss: 0.2074 - accuracy: 0.9144 - val_loss: 0.0748 - val_accuracy: 0.9764
Epoch 2/3
182/182 - 112s - loss: 0.0647 - accuracy: 0.9785 - val_loss: 0.0541 - val_accuracy: 0.9800
Epoch 3/3
182/182 - 111s - loss: 0.0519 - accuracy: 0.9826 - val_loss: 0.0470 - val_accuracy: 0.9832
<tensorflow.python.keras.callbacks.History at 0x7fa930e68d50>
```
We see that the training time per epoch is 114 seconds instead of 344 seconds previously. Also the train and validation accuracy is around 98% to start with. Please train the model for 30 epochs to get acuracy around 99.6%
Hence, we can say that using pre-trained models signifcantly improves the accuracy as well as reduces the training time.

**Some Additional Information**

The pre-trained model we fine-tuned to classify images of cats and dogs is called VGG16, which is the model that won the 2014 ImageNet competition.

In the ImageNet competition, multiple teams compete to build a model that best classifies images from the ImageNet library. The ImageNet library houses thousands of images belonging to 1000 different categories.

Note that dogs and cats were included in the ImageNet library from which VGG16 was originally trained. Therefore, the model has already learned the features of cats and dogs. Given this, the fine-tuning we've done on this model is very minimal. In future blogs, we'll do more involved fine-tuning and utilize transfer learning to classify completely new data than what was included in the original training set.

In this post, we have discussed only about Transfer Learning. However, we have not discussed about preprocessing the data that is required for transfer learning. For VGG-16 model, we have preprocessed the image in a certain way as it should be for VGG-16 model training. Further details about this is explained in the [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556.pdf) paper.

Please access the [GitHub](https://github.com/saptarshidatta96/Transfer-Learning-with-VGG-16) repository to view the code. 

© 2020 Saptarshi Datta. All Rights Reserved.
