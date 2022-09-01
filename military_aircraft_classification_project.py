#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.applications.vgg16 import VGG16  
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np

train_files_path = "airplanedataset/Train/"
test_files_path = "airplanedataset/Test/"
img = load_img(test_files_path + "B-52/3-1.jpg")

print(img_to_array(img).shape)

plt.imshow(img)
plt.show()

train_data = ImageDataGenerator().flow_from_directory(train_files_path,target_size = (224,224))
test_data = ImageDataGenerator().flow_from_directory(test_files_path,target_size = (224,224))

numberOfAirplaneTypes = 5  

vgg = VGG16()

vgg_layers = vgg.layers
print(vgg_layers)

vggmodel_layersize_tobe_used = len(vgg_layers) - 1

model = Sequential()
for i in range(vggmodel_layersize_tobe_used):
    model.add(vgg_layers[i])

for layers in model.layers:
    layers.trainable = False

model.add(Dense(numberOfAirplaneTypes, activation="softmax"))

print(model.summary())

model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])

batch_size = 4 

model.fit_generator(train_data,
                           steps_per_epoch=400//batch_size,
                           epochs= 3, 
                           validation_data=test_data,
                           validation_steps= 200//batch_size)

img = Image.open("f22.jpg").resize((224,224))

img = np.array(img)

img.shape

print(img.ndim)

img = img.reshape(-1,224,224,3)  

print(img.shape)
print(img.ndim)

img = preprocess_input(img)   

img_for_display = load_img("f22.jpg")
plt.imshow(img_for_display)
plt.show()

preds = model.predict(img)
preds

image_classes = ["A-10 Thunderbolt","Boeing B-52","Boeing E-3 Sentry","F-22 Raptor","KC-10 Extender"]

result = np.argmax(preds[0])
print(image_classes[result]) 

