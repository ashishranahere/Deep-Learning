#!/usr/bin/env python
# coding: utf-8

# In[743]:


import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[744]:


def read_data(path):
    
    train_path = path+"/train"
    test_path = path+"/test"
    validation_path = path+"/val"
    
    tr_data, test_data, val_data = [], [], []
    tr_out, test_out, val_out = [], [], []

    for i in os.listdir(train_path):
        try:
            for j in os.listdir(train_path+"/"+i):
                image = cv2.imread(train_path+"/"+i+"/"+j) #COLOR_BGR2RGB
                tr_data.append(cv2.resize(image, (224, 224)))
                tr_out.append(i)

            for j in os.listdir(test_path+"/"+i):
                image = cv2.imread(test_path+"/"+i+"/"+j)
                test_data.append(cv2.resize(image, (224, 224)))
                test_out.append(i)

            for j in os.listdir(validation_path+"/"+i):
                image = cv2.imread(validation_path+"/"+i+"/"+j)
                val_data.append(cv2.resize(image, (224, 224)))
                val_out.append(i)
        except:
            pass
                
    tr_data, test_data, val_data = np.array(tr_data), np.array(test_data), np.array(val_data)
    return tr_data, test_data, val_data, tr_out, test_out, val_out


# In[745]:


def label_data(x):
    label = []
    for i in x:
        if i =='car_side':
            label.append(0)
        if i =='hawksbill':
            label.append(1)
        if i =='kangaroo':
            label.append(2)
        if i =='scorpion':
            label.append(3)
        if i =='starfish':
            label.append(4)
    return label


# In[746]:


path = "/home/urvashi/Downloads/Group_5"
tr_data, test_data, val_data, tr_out, test_out, val_out = read_data(path)


# In[747]:


tr_label = np.array(label_data(tr_out))
val_label = np.array(label_data(val_out))
test_label = np.array(label_data(test_out))


# # Architecture 1

# In[8]:


model_a1 = models.Sequential()
model_a1.add(layers.Conv2D(filters=8, kernel_size=11, strides=(4,4) , padding='valid', activation='relu', input_shape=(224, 224, 3)))
model_a1.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))
model_a1.add(layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1,1) , padding='valid', activation='relu'))
model_a1.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

model_a1.add(layers.Flatten())
model_a1.add(layers.Dense(128, activation='relu'))
model_a1.add(layers.Dense(5, activation='softmax'))
model_a1.summary()


# In[9]:


model_a1.compile(optimizer='adam',  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=2,restore_best_weights=True)
history_a1 = model_a1.fit(np.array(tr_data), tr_label, validation_data=(val_data, val_label), epochs=1000 , callbacks=callback)


# In[10]:


plt.plot(history_a1.history['accuracy'], label='accuracy')
plt.plot(history_a1.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')


# In[11]:


val_loss, val_acc = model_a1.evaluate(val_data,  val_label, verbose=1)


# In[12]:


predictions = model_a1.predict(test_data, verbose=1)
p_class = np.argmax(predictions, axis=1)
cm = confusion_matrix(test_label, p_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["car_side","hawksbill","kangaroo","scorpion","starfish"])
disp.plot()
plt.xticks(rotation = 90)
plt.show()


# # Architecture 2 with Transfer Learning

# In[13]:


model_a2_trans = models.Sequential()
model_a2_trans.add(model_a1.layers[0])
model_a2_trans.add(model_a1.layers[1])
model_a2_trans.add(model_a1.layers[2])
model_a2_trans.add(model_a1.layers[3])

print(model_a2_trans.layers[3])

for layer in model_a2_trans.layers[:4]:
    layer.trainable = False

model_a2_trans.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1) , padding='valid', activation='relu'))
model_a2_trans.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

model_a2_trans.add(layers.Flatten())
model_a2_trans.add(layers.Dense(128, activation='relu'))
model_a2_trans.add(layers.Dense(5, activation='softmax'))
2

model_a2_trans.summary()
                                                                                    
                                                                                


# In[14]:


model_a2_trans.compile(optimizer='adam',  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=2,restore_best_weights=True)
history_a2_trans = model_a2_trans.fit(np.array(tr_data), tr_label,validation_data=(val_data, val_label), epochs=1000,  callbacks=callback)


# In[15]:


val_loss, val_acc = model_a2_trans.evaluate(val_data,  val_label, verbose=1)


# In[16]:


test_loss, test_acc = model_a2_trans.evaluate(test_data,  test_label, verbose=1)


# In[17]:


plt.plot(history_a2_trans.history['accuracy'], label='accuracy')
plt.plot(history_a2_trans.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')


# In[18]:


predictions = model_a2_trans.predict(test_data, verbose=1)
p_class = np.argmax(predictions, axis=1)
cm = confusion_matrix(test_label, p_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["car_side","hawksbill","kangaroo","scorpion","starfish"])
disp.plot()
plt.xticks(rotation = 90)
plt.show()


# # Architecture 2 without TL

# In[146]:


model_a2 = models.Sequential()
model_a2.add(layers.Conv2D(filters=8, kernel_size=11, strides=(4,4) , padding='valid', activation='relu', input_shape=(224, 224, 3)))
model_a2.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))
model_a2.add(layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1,1) , padding='valid', activation='relu'))
model_a2.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))
model_a2.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1) , padding='valid', activation='relu'))
model_a2.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

model_a2.add(layers.Flatten())
model_a2.add(layers.Dense(128, activation='relu'))
model_a2.add(layers.Dense(5, activation='softmax'))
model_a2.summary()


# In[150]:


model_a2.compile(optimizer='adam',  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=2,restore_best_weights=True)
history_a2 = model_a2.fit(np.array(tr_data), tr_label, validation_data=(val_data, val_label), epochs=1000,  callbacks=callback)


# In[151]:



val_loss, val_acc = model_a2.evaluate(val_data,  val_label, verbose=1)


# In[152]:


plt.plot(history_a2.history['accuracy'], label='accuracy')
plt.plot(history_a2.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')


# In[153]:


predictions = model_a2.predict(test_data, verbose=1)
p_class = np.argmax(predictions, axis=1)
cm = confusion_matrix(test_label, p_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["car_side","hawksbill","kangaroo","scorpion","starfish"])
disp.plot()
plt.xticks(rotation = 90)
plt.show()


# # Architecture 3 with Transfer Learning

# In[24]:


model_a3_trans = models.Sequential()
model_a3_trans.add(model_a1.layers[0])
model_a3_trans.add(model_a1.layers[1])
model_a3_trans.add(model_a1.layers[2])
model_a3_trans.add(model_a1.layers[3])


for layer in model_a3_trans.layers[:4]:
    layer.trainable = False

model_a3_trans.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1) , padding='valid', activation='relu'))    
model_a3_trans.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1) , padding='valid', activation='relu'))
model_a3_trans.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

model_a3_trans.add(layers.Flatten())
model_a3_trans.add(layers.Dense(128, activation='relu'))
model_a3_trans.add(layers.Dense(5, activation='softmax'))

# for layer in model_a2_trans.layers:
#     print(layer.trainable)

model_a3_trans.summary()
                                                                                    
                                                                                


# In[25]:


model_a3_trans.compile(optimizer='adam',  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=2,restore_best_weights=True)
history_a3_trans = model_a3_trans.fit(np.array(tr_data), tr_label, validation_data=(val_data, val_label), epochs=1000 ,  callbacks=callback)


# In[26]:


val_loss, val_acc = model_a3_trans.evaluate(val_data,  val_label, verbose=1)


# In[27]:


plt.plot(history_a3_trans.history['accuracy'], label='accuracy')
plt.plot(history_a3_trans.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')


# In[28]:


predictions = model_a3_trans.predict(test_data, verbose=1)
p_class = np.argmax(predictions, axis=1)
cm = confusion_matrix(test_label, p_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["car_side","hawksbill","kangaroo","scorpion","starfish"])
disp.plot()
plt.xticks(rotation = 90)
plt.show()


# # Architecture 3

# In[109]:


model_a3 = models.Sequential()
model_a3.add(layers.Conv2D(filters=8, kernel_size=11, strides=(4,4) , padding='valid', activation='relu', input_shape=(224, 224, 3)))
model_a3.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))
model_a3.add(layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1,1) , padding='valid', activation='relu'))
model_a3.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))
model_a3.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1) , padding='valid', activation='relu'))
model_a3.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1) , padding='valid', activation='relu'))
model_a3.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

model_a3.add(layers.Flatten())
model_a3.add(layers.Dense(128, activation='relu'))
model_a3.add(layers.Dense(5, activation='softmax'))
model_a3.summary()


# In[110]:


model_a3.compile(optimizer='adam',  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=2,restore_best_weights=True)
history_a3 = model_a3.fit(np.array(tr_data), tr_label, validation_data=(val_data, val_label), epochs=1000, callbacks=callback)


# In[111]:


val_loss, val_acc = model_a3.evaluate(val_data,  val_label, verbose=1)


# In[112]:


plt.plot(history_a3.history['accuracy'], label='accuracy')
plt.plot(history_a3.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')


# In[113]:


predictions = model_a3.predict(test_data, verbose=1)
p_class = np.argmax(predictions, axis=1)
cm = confusion_matrix(test_label, p_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["car_side","hawksbill","kangaroo","scorpion","starfish"])
disp.plot()
plt.xticks(rotation = 90)
plt.show()


# # All 8 feature maps from the first convolutional layers and a selected 8 feature maps from the remaining convolutional layers of the best architecture.

# In[34]:


train_img = np.array(tr_data[52])
plt.imshow(train_img)
train_img.shape


# In[36]:


for i in range(len(model_a2.layers)):
    layer = model_a2.layers[i]
    if 'conv' not in layer.name:
        continue    
    print(i , layer.name , layer.output.shape)


# In[38]:


model_1 = tf.keras.models.Model(inputs=model_a2.inputs , outputs=model_a2.layers[0].output)


# In[39]:


#calculating features_map
features = model_1.predict(np.array(tr_data[52:53]))

fig = plt.figure(figsize=(30,30))
for i in range(1,features.shape[3]+1):
    plt.subplot(8,8,i)
    plt.imshow(features[0,:,:,i-1])
    
plt.show()


# In[41]:


model_2 = tf.keras.models.Model(inputs=model_a2.inputs , outputs=model_a2.layers[2].output)


# In[42]:


#calculating features_map
features = model_2.predict(np.array(tr_data[52:53]))

fig = plt.figure(figsize=(30,30))
for i in range(1,features.shape[3]+1):
    plt.subplot(8,8,i)
    plt.imshow(features[0,:,:,i-1])
    
plt.show()


# In[43]:


model_3 = tf.keras.models.Model(inputs=model_a2.inputs , outputs=model_a2.layers[4].output)


# In[44]:


#calculating features_map
features = model_3.predict(np.array(tr_data[52:53]))

fig = plt.figure(figsize=(30,30))
for i in range(1,features.shape[3]+1):
    plt.subplot(8,8,i)
    plt.imshow(features[0,:,:,i-1])
    
plt.show()


# # Visualizing Patches which Maximally Activate a Neuron
# 

# In[742]:


image = cv2.imread('/home/urvashi/Downloads/Group_5/train/kangaroo/image_0007.jpg')
image = cv2.resize(image, (224, 224))
plt.imshow(image)

image.shape


# In[709]:


test = np.expand_dims(image, axis=0)
test.shape


# In[710]:


model_a2.summary()


# In[712]:


def get_feature(layer_name, test):
  layer_outputs = [layer.output for layer in model_a2.layers if layer.name == layer_name]
  activation_model = tf.keras.models.Model(inputs=model_a2.input, outputs=layer_outputs)
  return activation_model.predict(test)


# In[713]:


last_conv = get_feature('conv2d_43' , test)


# In[714]:


for i in range(25):
  plt.subplot(5, 5, i+1)
  plt.imshow(np.squeeze(last_conv[:, :, :, i]))
plt.show()


# In[715]:


max_pos = np.argmax(np.squeeze(last_conv[:, :, :, 1]))
max_pos


# In[716]:


np.ndarray.flatten(np.squeeze(last_conv[:, :, :, i]))[max_pos], np.amax(np.squeeze(last_conv[:, :, :, 1]))


# In[717]:


conv2d_11 = get_feature('conv2d_5', test)


# In[718]:


#max_pos = np.argmax(np.squeeze(last_conv[:, :, :, 1]))
max_value = np.max(np.squeeze(last_conv[:, :, :, 1]))
max_pos = np.where(np.squeeze(last_conv[:, :, :, 1]) == max_value)
max_pos = (max_pos[0][0], max_pos[1][0])
max_pos


# In[719]:


def trace_patch(kernel_size, stride, max_pos):
    '''
    This function returns the positions of the pixels in the previous feature map that correspond to the output pixel at max_pos.
    kernel_size: size of the convolution kernel
    stride: stride of the convolution
    max_pos: (x,y) coordinates of the output pixel in the current feature map
    '''
    padding = 0
    input_positions = []
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = (max_pos[0] * stride) - padding + j
            y = (max_pos[1] * stride) - padding + i
            input_positions.append((x,y))
    
    return input_positions


# In[720]:


# Strides = 1 and padding = 1 (same) in the second last Conv Layer
# patch1 = trace_patch(max_pos)
patch1 = trace_patch(3,2, max_pos) 


# In[721]:


patch1


# In[722]:


patch2 = []
for i in patch1:
  patch2.extend(trace_patch(5,1,i))

patch2 = list(set(patch2)) #unique positions only
print(len(patch2))


# In[723]:


patch3 = []
for i in patch2:
  patch3.extend(trace_patch(3, 2, i))

patch3 = list(set(patch3)) #unique positions only
print(patch3)


# In[724]:


len(patch3)


# In[725]:


patch4 = []
for i in patch3:
  patch4.extend(trace_patch(11,4,i))

patch4 = list(set(patch4)) #unique positions only
# print(patch4)


# In[749]:


# image = cv2.imread('C:/Users/shilp/OneDrive/Documents/CS671/A5_data/train/4/image_0077.jpg')
image = cv2.imread('/home/urvashi/Downloads/Group_5/train/kangaroo/image_0007.jpg')
image = cv2.resize(image, (224, 224))
# plt.imshow(image, cmap='gray')

image.shape


# In[750]:


def find_extreme_pixels(pixel_points):
    # initialize variables to store extreme pixels
    bottom_left = pixel_points[0]
    bottom_right = pixel_points[0]
    top_left = pixel_points[0]
    top_right = pixel_points[0]

    # loop through all the pixels to find the extreme pixels
    for pixel in pixel_points:
        if pixel[1] > bottom_left[1] or (pixel[1] == bottom_left[1] and pixel[0] < bottom_left[0]):
            bottom_left = pixel
        if pixel[1] > bottom_right[1] or (pixel[1] == bottom_right[1] and pixel[0] > bottom_right[0]):
            bottom_right = pixel
        if pixel[1] < top_left[1] or (pixel[1] == top_left[1] and pixel[0] < top_left[0]):
            top_left = pixel
        if pixel[1] < top_right[1] or (pixel[1] == top_right[1] and pixel[0] > top_right[0]):
            top_right = pixel

    # return the extreme pixels
    return bottom_left, bottom_right, top_left, top_right


# In[751]:


bottom_left, bottom_right, top_left, top_right = find_extreme_pixels(patch4)
print(bottom_left,bottom_right,top_left,top_right)


# In[752]:


# Determine the rectangle dimensions
width = bottom_right[0] - top_left[0]
height = bottom_right[1] - top_left[1]


# In[753]:


# Create a copy of the image to draw the rectangle on
image_with_rectangle = image.copy()


# In[754]:


# Extract the rectangular region from the original image
rectangular_region = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
plt.imshow(rectangular_region)


# In[755]:


# Draw the rectangle on the copied image
cv2.rectangle(image_with_rectangle, top_left, bottom_right, (0, 254, 0), thickness=2)


# In[756]:


plt.imshow(image_with_rectangle)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




