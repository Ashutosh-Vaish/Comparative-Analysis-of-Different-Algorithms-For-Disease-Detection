from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# resizing all images
IMAGE_SIZE = [224, 224]

train_dir = "../DiseaseDetection/chest_xray/train/"
test_dir = "../DiseaseDetection/chest_xray/test/"

# Import the Resnet50 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights
resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False,classes=1000)

# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False
    
 # useful for getting number of output classes
folders = glob('../DiseaseDetection/chest_xray/test/*')

# our layers 
x = Flatten()(resnet.output)

prediction = Dense(1, activation='softmax')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)

model.summary()

model.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Using the Image Data Generator to import the images from the dataset
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Making sure to provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('../DiseaseDetection/chest_xray/train/',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('../DiseaseDetection/val',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')

# fitting the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=20,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
