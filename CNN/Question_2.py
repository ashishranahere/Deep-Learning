#!/usr/bin/env python
# coding: utf-8

# # Visualizing Patches: VGG19

# In[389]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras import backend as K
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input


# In[390]:


import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


# # Read images and Resize image to 224x224

# In[391]:


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


# In[392]:


path = "/home/urvashi/Downloads/Group_5"
train_data, test_data, val_data, tr_out, test_out, val_out = read_data(path)


# In[393]:


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


# In[394]:


train_label = np.array(label_data(tr_out))
val_label = np.array(label_data(val_out))
test_label = np.array(label_data(test_out))


# In[395]:


# Load the VGG19 model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the weights of the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add a new classification layer with 5 output nodes
x = base_model.output
x = Flatten()(x)
x = Dense(5, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping

# Define the early stopping criteria
callback = EarlyStopping(monitor="val_loss", min_delta=0.0001, verbose=0, restore_best_weights=True, patience=2)

# Train the model
model.fit(train_data, train_label, epochs=10000, batch_size=32, callbacks=[callback], validation_data=(val_data, val_label))

# Evaluate the model on the training set
train_loss, train_acc = model.evaluate(train_data, train_label)
print('Training accuracy:', train_acc)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_data, val_label)
print('Validation accuracy:', val_acc)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_data, test_label)
print('Test accuracy:', test_acc)


# In[ ]:


model.summary()


# ### Sample Test Image

# In[ ]:


image = cv2.imread('/home/urvashi/Downloads/Group_5/train/starfish/image_0076.jpg')
image = cv2.resize(image, (224, 224))
plt.imshow(image)

image.shape


# ### Testing

# In[ ]:


test = np.expand_dims(image, axis=0)
test.shape


# ### Traceback

# In[ ]:


def get_feature(layer_name, test):
  layer_outputs = [layer.output for layer in model.layers if layer.name == layer_name]
  activation_model = Model(inputs=model.input, outputs=layer_outputs)
  return activation_model.predict(test)


# ##### Plotting one example

# In[ ]:


last_conv = get_feature('block5_conv4' , test)


# In[ ]:


for i in range(25):
  plt.subplot(5, 5, i+1)
  plt.imshow(np.squeeze(last_conv[:, :, :, i]))
plt.show()


# ##### Finding Maximally activated neuron in final conv layer

# In[ ]:


max_pos = np.argmax(np.squeeze(last_conv[:, :, :, 1]))
max_pos


# In[ ]:


np.ndarray.flatten(np.squeeze(last_conv[:, :, :, i]))[max_pos], np.amax(np.squeeze(last_conv[:, :, :, 1]))


# ##### Getting features of each layer w/o gradients

# In[ ]:


block5conv4 = get_feature('block5_conv4', test)


# ##### Back tracing last three conv layers (Note the strides and padding)

# In[ ]:


def trace_patch(kernel_size, stride, max_pos):
    '''
    This function returns the positions of the pixels in the previous feature map that correspond to the output pixel at max_pos.
    kernel_size: size of the convolution kernel
    stride: stride of the convolution
    max_pos: (x,y) coordinates of the output pixel in the current feature map
    '''
    padding = (kernel_size - 1) // 2 # compute the padding size needed to keep the spatial dimensions unchanged
    input_positions = []
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = (max_pos[0] * stride) - padding + j
            y = (max_pos[1] * stride) - padding + i
            input_positions.append((x,y))
    
    return input_positions


# In[ ]:


#max_pos = np.argmax(np.squeeze(last_conv[:, :, :, 1]))
max_value = np.max(np.squeeze(last_conv[:, :, :, 1]))
print(max_value)
max_pos = np.where(np.squeeze(last_conv[:, :, :, 1]) == max_value)
max_pos = (max_pos[0][0], max_pos[1][0])
max_pos


# In[ ]:


patch1 = trace_patch(3,1,max_pos)


# In[ ]:


patch1


# In[ ]:


patch2 = []
for i in patch1:
  patch2.extend(trace_patch(3,1,i))

patch2 = list(set(patch2)) #unique positions only
print(patch2)


# In[ ]:


patch3 = []
for i in patch2:
  patch3.extend(trace_patch(2,2,i))

patch3 = list(set(patch3)) #unique positions only
print(patch3)


# In[ ]:


patch4 = []
for i in patch3:
  patch4.extend(trace_patch(3,1,i))

patch4 = list(set(patch4)) #unique positions only
# print(patch4)


# In[ ]:


patch5 = []
for i in patch4:
  patch5.extend(trace_patch(3,1,i))

patch5 = list(set(patch5)) #unique positions only
# print(patch5)


# In[ ]:


patch6 = []
for i in patch5:
  patch6.extend(trace_patch(3,1,i))

patch6 = list(set(patch6)) #unique positions only
# print(patch6)


# In[ ]:


patch7 = []
for i in patch6:
  patch7.extend(trace_patch(3,1,i))

patch7 = list(set(patch7)) #unique positions only
# print(patch7)


# In[ ]:


patch8 = []
for i in patch7:
  patch8.extend(trace_patch(2,2,i))

patch8 = list(set(patch8)) #unique positions only


# In[ ]:


patch9 = []
for i in patch8:
  patch9.extend(trace_patch(3,1,i))

patch9 = list(set(patch9)) #unique positions only

patch10 = []
for i in patch9:
  patch10.extend(trace_patch(3,1,i))

patch10 = list(set(patch10)) #unique positions only


# In[ ]:


patch11 = []
for i in patch10:
  patch11.extend(trace_patch(3,1,i))

patch11 = list(set(patch11)) #unique positions only


# In[ ]:


patch12 = []
for i in patch11:
  patch12.extend(trace_patch(3,1,i))

patch12 = list(set(patch12)) #unique positions only

patch13 = []
for i in patch12:
  patch13.extend(trace_patch(2,2,i))

patch13 = list(set(patch13)) #unique positions only


# In[ ]:


patch14 = []
for i in patch13:
  patch14.extend(trace_patch(3,1,i))

patch14 = list(set(patch14)) #unique positions only
# print(patch7)


# In[ ]:


patch15 = []
for i in patch14:
  patch15.extend(trace_patch(3,1,i))

patch15 = list(set(patch15)) #unique positions only

patch16 = []
for i in patch15:
  patch16.extend(trace_patch(2,2,i))

patch16 = list(set(patch16)) #unique positions only


# In[ ]:


patch17 = []
for i in patch16:
  patch17.extend(trace_patch(3,1,i))

patch17 = list(set(patch17)) #unique positions only

patch18 = []
for i in patch17:
  patch18.extend(trace_patch(3,1,i))

patch18 = list(set(patch18)) #unique positions only


# In[ ]:


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


# In[ ]:


bottom_left, bottom_right, top_left, top_right = find_extreme_pixels(patch16)
print(bottom_left,bottom_right,top_left,top_right)


# In[ ]:


top_left = (max(0, min(top_left[0], 223)), max(0, min(top_left[1], 223)))
top_right = (max(0, min(top_right[0], 223)), max(0, min(top_right[1], 223)))
bottom_left = (max(0, min(bottom_left[0], 223)), max(0, min(bottom_left[1], 223)))
bottom_right = (max(0, min(bottom_right[0], 223)), max(0, min(bottom_right[1], 223)))


# In[ ]:


# Determine the rectangle dimensions
width = bottom_right[0] - top_left[0]
height = bottom_right[1] - top_left[1]


# In[ ]:


# Create a copy of the image to draw the rectangle on
image_with_rectangle = image.copy()


# In[ ]:


# Extract the rectangular region from the original image
rectangular_region = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
plt.imshow(rectangular_region)


# In[ ]:


# Draw the rectangle on the copied image
cv2.rectangle(image_with_rectangle, top_left, bottom_right, (0, 254, 0), thickness=2)


# In[ ]:


plt.imshow(image_with_rectangle)


# # Guided Backpropagagtion

# In[432]:


@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad

IMAGE_SIZE = [224, 224]



class GuidedBackprop:
    def __init__(self,model, layerName=None):
        self.model = model
        self.layerName = layerName
        self.gbModel = self.build_guided_model()
        
        if self.layerName == None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")

    def build_guided_model(self):
        print(self.model.get_layer(self.layerName).output.shape)
        dum = self.model.get_layer(self.layerName).output[:, :, :, 1]
        #dum = tf.expand_dims(dum, axis=0)
        print("hey",dum)
        gbModel = Model(
            inputs = [self.model.inputs],
            outputs = [dum]
        )
        layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer,"activation")]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu
        
        return gbModel
    
    def guided_backprop(self, images, upsample_size,class_index=None):
        with tf.GradientTape() as tape:
#             images=cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            if class_index is None:
                outputs = self.gbModel(inputs)
            else:
                outputs = self.gbModel(inputs)[:,class_index]

        grads = tape.gradient(outputs, inputs)[0]

        saliency = cv2.resize(np.asarray(grads), upsample_size)
#         saliency=tf.image.grayscale_to_rgb(tf.cast(saliency, tf.uint8))
        return saliency


# In[433]:


def show_BP(GuidedBP, im_ls, n=3):

    plt.subplots(figsize=(30, 10*n))
    k=1
    for i in range(n):
        img = image.load_img(im_ls[i],target_size=(224,224))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Show original image
        plt.subplot(n,3,k)
        plt.imshow(img,cmap="gray")
        plt.title("Input image", fontsize=20)
        plt.axis("off")
        
        x = image.img_to_array(img)

        upsample_size = (x.shape[1],x.shape[0])
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)


        plt.subplot(n,3,k+1)
        gb = GuidedBP.guided_backprop(x, upsample_size,class_index=7)
#         print(gb)
        gb_viz = np.dstack((
            gb[:, :, 0],
            gb[:, :, 1],
            gb[:, :, 2],))
        gb_viz -= np.min(gb_viz)
        gb_viz /= gb_viz.max()
        
        scal = MinMaxScaler()
        
        plt.imshow(gb_viz,cmap="gray")
        plt.title("Guided Backprop", fontsize=20)
        plt.axis("off")

        plt.subplot(n,3,k+2)
        gb[:, :, 0] = scal.fit_transform(gb[:, :, 0])
        gb[:, :, 1] = scal.fit_transform(gb[:, :, 1])
        gb[:, :, 2] = scal.fit_transform(gb[:, :, 2])
        
        
#         plt.imshow(gb,cmap="gray")
#         plt.title("MinMaxScaled", fontsize=20)
        plt.axis("off")
        k += 3
    plt.show()


# In[ ]:


guidedBP = GuidedBackprop(model = model, layerName="block5_conv4")
img_path = ['C:/Users/Ankit Mehra/Downloads/Group_5/Group_5/train/starfish/image_0008.jpg', 'C:/Users/Ankit Mehra/Downloads/Group_5/Group_5/train/scorpion/image_0056.jpg', 'C:/Users/Ankit Mehra/Downloads/Group_5/Group_5/train/kangaroo/image_0045.jpg','C:/Users/Ankit Mehra/Downloads/Group_5/Group_5/train/hawksbill/image_0092.jpg','C:/Users/Ankit Mehra/Downloads/Group_5/Group_5/train/car_side/image_0007.jpg']
show_BP(guidedBP, img_path , n=5, )


# # Grad-CAM

# In[434]:


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# In[435]:


last_conv_layer_name = "block5_conv4"
img_path = "C:/Users/Ankit Mehra/Downloads/Group_5/Group_5/train/starfish/image_0076.jpg"
image = cv2.imread(img_path)
img = cv2.resize(image, (224, 224))
plt.imshow(img)


# In[ ]:


# Prepare image
img_size = (224, 224)
img1 = np.expand_dims(img,axis=0)
img_array = preprocess_input(img1)

# Make model
# model = model_builder(weights="imagenet")

# Remove last layer's softmax
model.layers[-1].activation = None

# Print what the top predicted class is
preds = model.predict(img_array)
# print("Predicted:", decode_predictions(preds, top=1)[0])

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name,pred_index=3)

# Display heatmap
plt.matshow(heatmap)
plt.show()


# In[ ]:


def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
#     img = keras.preprocessing.image.load_img(img_path)
#     img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
#     plt.imshow(np.array(tr_data[212]))
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


save_and_display_gradcam(img, heatmap)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




