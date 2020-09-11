---
published: true
---
In the last [post](https://git.saptarshidatta.in/Transfer-Learning-with-VGG-16/), we applied Transfer Learning in the VGG-16 Model with the Cat vs Dogs data set. However the application was minimal as we only changed the last output layer from a 'softmax' actiavted outpur to a 'sigmoid' activated output. Additionally, the VGG-16 model was already trained on the ImageNet Data, which originally had imaged of cats and dogs. We jist trained the last dense layer which predicted whether the image is og a cat or a dog.

However, in this exercise we shall again apply transfer learning to predict the Numeric Sign Language. We will be applying MobileNet Model and shall modify the model and then fine tune it to suit our requirements. But before that, let's discuss a bit about the MobileNet Model.

## MobileNet Model
MobileNets are a class of small, low-latency and low-power model that can be used for classification, detection, and other common tasks convolutional neural networks are good for. Because of their small size, these are considered great deep learning models to be used on mobile devices.

The MobileNet is about 17 MB in size and has just 4.2 million parameters as compared to the VGG-16 model which has a size of 534 MB and around 138 Million parameters.

However due to their smaller size and faster performance than other networks, MobileNets aren’t as accurate as the other large, resource-heavy models. However they still actually perform very well, with really only a relatively small reduction in accuracy.

Please go through the [MobileNet Paper](https://arxiv.org/pdf/1704.04861.pdf) which elaborates further regarding the tradeoff between accuracy and size.

Having discussed the MobileNet model to some extent, let move ahead to other sections. Alright, let's jump into the code!

## Data Preparation
We have used the Sign Language Digits dataset from [GitHub](https://github.com/ardamavi/Sign-Language-Digits-Dataset). The data is located in corresponding folders ranging from 0-9, however we will use a script to divide the data into train, test and valid datasets.

```
os.chdir('/content/gdrive/My Drive/Sign-Language-Digits-Dataset/Dataset')
if os.path.isdir('train/0/') is False: 
    os.mkdir('train')
    os.mkdir('valid')
    os.mkdir('test')

    for i in range(0, 10):
        shutil.move(f'{i}', 'train')
        os.mkdir(f'valid/{i}')
        os.mkdir(f'test/{i}')

        valid_samples = random.sample(os.listdir(f'train/{i}'), 30)
        for j in valid_samples:
            shutil.move(f'train/{i}/{j}', f'valid/{i}')

        test_samples = random.sample(os.listdir(f'train/{i}'), 5)
        for k in test_samples:
            shutil.move(f'train/{i}/{k}', f'test/{i}')

os.chdir('../..')
```

So what we are basically doing in the above script is at first checking whether a train folder already exists, if not, we are creating a train/test/valid folders. Then, we will move all the images corresponding to a particular class-folder from the main folder to the corresponding class-folder inside the train folder, and at the same time creating new class-folder inside the valid and test folders. Then we randomly move 30 and 5 images from each class-folder inside train folder to the corresponding class-folders inside valid and test folders that were created before.
We run the entire process in a loop iterating over class-folders ranging from 0-9.

Next, we preprocess the train, valid and test data in a fashion the MobileNet model expects(MobileNet expects images to be scales between [-1,1] rather than [0, 225]. We set the batch size to 10 and the target image size to (224, 224) since that is the image size that MobileNet expects. We set shuffle to False for the test batch, so that later we can plot our results on to a confusion Matrix.

```
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)
```

## Fine Tuning the MobileNet Model

We will import the MobileNet Model just as we imported the VGG-16 Model. Then we will drop the last 5 layers forom the model and add a dense layer with softmax activation predicting 10 classes ranging from 0-9. Later we freeze all the layers except the last 23 layers (A MobileNet Model has 88 Layers). The choice of the number 23 is based upon personal choice and some experimentation. It was found out that if we train the last 23 layers, we get some really good results.
Please note that it is a significant deviation from the last VGG-16 Model training, where we only trained the last output layer.

```
mobile = tf.keras.applications.mobilenet.MobileNet()
x = mobile.layers[-6].output
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=mobile.input, outputs=predictions)
for layer in model.layers[:-23]:
    layer.trainable = False
```

Now, we will compile and train the model for 30 epochs and our model scores 100% valodation accuracy and 98.89% train accuracy. This shows taht our model is generalising well.

```
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, steps_per_epoch=18, validation_data=valid_batches, validation_steps=3, epochs=30, verbose=2)
```

Here are the accuracies for the last 5 epochs:

```
Epoch 25/30
18/18 - 1s - loss: 0.0449 - accuracy: 0.9944 - val_loss: 0.0444 - val_accuracy: 1.0000
Epoch 26/30
18/18 - 2s - loss: 0.0510 - accuracy: 0.9944 - val_loss: 0.0346 - val_accuracy: 1.0000
Epoch 27/30
18/18 - 1s - loss: 0.0329 - accuracy: 1.0000 - val_loss: 0.0564 - val_accuracy: 1.0000
Epoch 28/30
18/18 - 2s - loss: 0.0312 - accuracy: 1.0000 - val_loss: 0.0276 - val_accuracy: 1.0000
Epoch 29/30
18/18 - 2s - loss: 0.0427 - accuracy: 0.9944 - val_loss: 0.0664 - val_accuracy: 0.9667
Epoch 30/30
18/18 - 1s - loss: 0.0344 - accuracy: 0.9889 - val_loss: 0.0609 - val_accuracy: 1.0000
<tensorflow.python.keras.callbacks.History at 0x7fd299fcbb00>
```

## Prediction on the Test Batch

Now, let's test the model on the test batch and plot the confusion matrix.

```
predictions = model.predict(x=test_batches, steps=5, verbose=0)
cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))
cm_plot_labels = ['0','1','2','3','4','5','6','7','8','9']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

Confusion matrix, without normalization
[[5 0 0 0 0 0 0 0 0 0]
 [0 5 0 0 0 0 0 0 0 0]
 [0 0 5 0 0 0 0 0 0 0]
 [0 0 0 5 0 0 0 0 0 0]
 [0 0 0 0 5 0 0 0 0 0]
 [0 0 0 0 0 5 0 0 0 0]
 [0 0 0 0 0 0 5 0 0 0]
 [0 0 0 0 0 0 0 4 1 0]
 [0 0 0 0 0 0 0 0 5 0]
 [0 0 0 0 0 0 0 0 0 5]]
```

Our model has performed excellent on the Test batch with only one error.
Please access the [GitHub](https://github.com/saptarshidatta96/Sign-Language-Digits-Prediction) Repository to access the Python Notebook associated with this exercise.
