from keras.models import load_model
from dataflow import flow_generator
from dir import get_val_case_dir
import cv2
import numpy as np
import tensorflow as tf

with tf.device('/device:CPU:0'):
    #Settings
    case = '3'
    image_size = 160
    batch_size = 8
    epochs = 20
    net = 'DenseNet121'
    IMG_SHAPE = (image_size, image_size, 3)
    fine_tune_at = 100
    mode = 'categorical' 
    function = 'categorical_crossentropy'

    #binary or categorical
    if case == '1':
        classes = 2
    else:
        classes = 3


    val_dir = get_val_case_dir(case)

    #model = load_model('models/DenseNet121-3.h5')

    model = load_model('models/MobileNetV2-3.h5')


    validation_flow = flow_generator(directory=val_dir, 
                    image_size=image_size,
                    batch_size=batch_size,
                    mode=mode)

    validation_steps = validation_flow.n / batch_size

    evaluate = model.evaluate_generator(validation_flow, validation_steps, workers=1)

    print(evaluate)