

import tensorflow as tf
import os
import numpy as np
from keras.preprocessing.image import load_img,img_to_array,array_to_img
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback,EarlyStopping

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle.json

!kaggle datasets download -d vipoooool/new-plant-diseases-dataset

from zipfile import ZipFile
with ZipFile('/content/new-plant-diseases-dataset.zip', 'r') as zipObj: zipObj.extractall()

import os
import numpy as np
from keras.preprocessing.image import load_img,img_to_array,array_to_img
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback,EarlyStopping


n_of_image,label_name = 100,['Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
         'Cherry healthy','Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight','Corn healthy',
         'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy','Peach Bacterial spot','Peach healthy', 'Pepper bell Bacterial spot',
         'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy',
         'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
         'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

img,label,img_size = [],[],(150,150)

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Apple___Apple_scab'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(0) # Apple___Apple_scab

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Apple___Black_rot'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(1) # Apple___Black_rot

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Apple___Cedar_apple_rust'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(2) # Apple___Cedar_apple_rust

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Apple___healthy'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(3) # Apple___healthy

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Cherry_(including_sour)___Powdery_mildew'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(4) # Cherry_(including_sour)___Powdery_mildew

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Cherry_(including_sour)___healthy'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(5) #Cherry_(including_sour)___healthy

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(6) # Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Corn_(maize)___Common_rust_'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(7) # Corn_(maize)___Common_rust_

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Corn_(maize)___Northern_Leaf_Blight'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(8) # Corn_(maize)___Northern_Leaf_Blight

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Corn_(maize)___healthy'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(9) # Corn_(maize)___healthy

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Grape___Black_rot'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(10) # Grape___Black_rot

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Grape___Esca_(Black_Measles)'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(11) # Grape___Esca_(Black_Measles)

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(12) # Grape___Leaf_blight_(Isariopsis_Leaf_Spot)

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Grape___healthy'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(13) # Grape___healthy

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Peach___Bacterial_spot'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(14) # Peach___Bacterial_spot

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Peach___healthy'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(15) # Peach___healthy

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Pepper,_bell___Bacterial_spot'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(16) # Pepper,_bell___Bacterial_spot

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Pepper,_bell___healthy'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(17) # Pepper,_bell___healthy

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Potato___Early_blight'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(18) # Potato___Early_blight

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Potato___Late_blight'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(19) # Potato___Late_blight

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Potato___healthy'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(20) # Potato___healthy

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Strawberry___Leaf_scorch'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(21) # Strawberry___Leaf_scorch

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Strawberry___healthy'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(22) # Strawberry___healthy

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Bacterial_spot'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(23) # Tomato___Bacterial_spot

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Early_blight'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(24) # Tomato___Early_blight

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Late_blight'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(25) # Tomato___Late_blight

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Leaf_Mold'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(26) # Tomato___Leaf_Mold

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Septoria_leaf_spot'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(27) # Tomato___Septoria_leaf_spot

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Spider_mites Two-spotted_spider_mite'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(28) # Tomato___Spider_mites Two-spotted_spider_mite

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Target_Spot'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(29) # Tomato___Target_Spot

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Tomato_Yellow_Leaf_Curl_Virus'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(30) # Tomato___Tomato_Yellow_Leaf_Curl_Virus

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Tomato_mosaic_virus'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(31) # Tomato___Tomato_mosaic_virus

path_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___healthy'
os.chdir(path_dir)
img_path_list = os.listdir(path_dir)
for len_no,img_path in enumerate(img_path_list):
  if len_no == n_of_image:break
  else:
    img.append(img_to_array(load_img(img_path,target_size=img_size))/255)
    label.append(32) # Tomato___healthy

img,label = np.array(img),np.array(label)

import tensorflow as tf
IMG_SHAPE = img_size + (3,)
vgg = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)

vgg.summary()

vgg.trainable = False

import pandas as pd

pd.set_option('max_colwidth', None)

layer_vgg = [(layer, layer.name, layer.trainable) for layer in vgg.layers]
pd.DataFrame(layer_vgg, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

average_pool = tf.keras.layers.GlobalAveragePooling2D()

prediction = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(33,activation='softmax')
])


tl_model = tf.keras.Sequential([
  vgg,
  average_pool,
  prediction,
  ])

tl_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

img,img_test,label,label_test = train_test_split(img,label,test_size=0.25,random_state=10)

os.chdir('/content')

class mycallback(Callback):
  def on_train_end(self,epoch,log={}):
    if (log.get('val_accuracy') >= '0.90'):
      print('Reached 90.9% accuracy so cancelling training')
      self.model.stop_training = True

castom_call = mycallback()

early_stop = EarlyStopping(monitor='val_accuracy', patience=3)


tl_model.fit(img,to_categorical(label), epochs=5, validation_data=(img_test,to_categorical(label_test)) )
            #,callbacks = [castom_call,early_stop])

#tl_model.save('Leaf Deases(96,88).h5')

for i in range(50):
    plt.imshow(img_test[i])
    plt.ylabel(label_test[i])
    img = img_test[i]
    pr = tl_model.predict(img.reshape((1,)+img.shape))
    plt.xlabel(np.argmax(pr))
    plt.show()
    plt.close()

