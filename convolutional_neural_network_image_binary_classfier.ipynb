import tensorflow as tf
#image data generator class
from keras.preprocessing.image   import ImageDataGenerator

# apply transformation on all the images. 
# rescale -> feature scaling 
# ranges to avoid overfitting and perform flip horizontal etc
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


train_generator = train_datagen.flow_from_directory(
        'dataset/testing_set',
        target_size=(64, 64), # image size to resize
        batch_size=32, # size of batchs ie images in a batch. 
        class_mode='binary') # class = binary or categorical


test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


test_generator = test_datagen.flow_from_directory(
        'dataset/testing_set',
        target_size=(64, 64), # image size to resize
        batch_size=32, # size of batchs ie images in a batch. 
        class_mode='binary') # class = binary or categorical

#initialize
cnn = tf.keras.models.Sequential()
# Convolute
#kernel =3 by 3 
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=[64,64,1]))
#pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        
    
# 2nd Convolution layer
#kernel =3 by 3 
#https://keras.io/api/layers/convolution_layers/convolution2d/
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
#pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#flattening
cnn.add(tf.keras.layers.Flatten())

#apply ANN
#units = hidden neurons, activation = relu
cnn.add(tf.keras.layers.Dense(units=122, activation="relu"))

#https://keras.io/api/layers/core_layers/dense/
#units = hidden neurons, activation = relu
cnn.add(tf.keras.layers.Dense(units=1, activation="sigmoid")) # for multi class softmax activation

#compile cnn loss is better than rms
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

#train model on cnn
cnn.fit(x = training_set, validation_data=test_set, epochs=25)

#predict
import numpy as np
from keras.preprocessing import image
test_image = image.load_img("dataset/single_prediction/class1_or_class2",target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
if result[0][0] == 1:
    prediction="class1"
else:
    prediction="class2"
    
    
print(prediction)    
