import numpy as np
from tensorflow import keras
from sklearn.svm import SVC
from dir import get_train_case_dir, get_val_case_dir
from dataflow import flow_generator_def

#settings
case = '1'
classes = 2
image_size = 160
batch_size = 1
IMG_SHAPE = (image_size, image_size, 3)

#load directories
train_dir = get_train_case_dir(case)
val_dir = get_val_case_dir(case)

#load model
base_model = keras.applications.DenseNet121(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

model = keras.Sequential([
    base_model,
    keras.layers.AveragePooling2D((5, 5), name='avg_pool'),
])

#generate data flow to extract training and validation features
train_gen = flow_generator_def(directory=train_dir, image_size=image_size, batch_size=batch_size)
train_features = model.predict_generator(train_gen, train_gen.samples / batch_size)
x_train = np.reshape(train_features, (-1, 1024))
y_train = train_gen.classes

val_gen = flow_generator_def(directory=val_dir, image_size=image_size, batch_size=batch_size)
val_features = model.predict_generator(val_gen, val_gen.samples / batch_size)
val_label = val_gen.classes
x_val = np.reshape(val_features, (-1, 1024))
y_val = val_gen.classes

print('training: {} {}'.format(x_train.size, y_train.size))
print('validation: {} {}'.format(x_val.size, y_val.size))

svm = SVC(kernel='rbf')

svm.fit(x_train, y_train)

print('fit done')

train_score = svm.score(x_train, y_train)

print('train score = {}'.format(train_score))

val_score = svm.score(x_val, y_val)

print('val score = {}'.format(val_score))
