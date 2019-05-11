from keras.optimizers import Adam
from dataflow import flow_generator_def
from dir import get_train_case_dir, get_val_case_dir
from DenseNet121 import load_DenseNet121
import numpy as np
from pprint import pprint

def main():
    print('extracting features...')

    case = '2'
    classes = 3
    image_size = 224
    batch_size = 8
    optimizer = Adam(lr=1e-3)
    net = 'DenseNet121-flat'
    IMG_SHAPE = (image_size, image_size, 3)

    #binary or categorical
    if case == '1':
        mode = 'binary'
        function = 'binary_crossentropy'
    else:
        mode = 'categorical'
        function = 'categorical_crossentropy'

    model = load_DenseNet121(input_shape=IMG_SHAPE, classes=classes, optimizer=optimizer, function=function, include_top=False, compile=True, verbose=0, trainable=False)
    extract_features(model, image_size, batch_size, case, mode)

def extract_features(model, image_size, batch_size, case, mode):
    train_dir = get_train_case_dir(case)
    val_dir = get_val_case_dir(case)
    
    train_gen = flow_generator_def(directory=train_dir, image_size=image_size, batch_size=batch_size, mode=mode)
    train_features = model.predict_generator(train_gen, train_gen.samples / batch_size)
    train_label = train_gen.classes
    train_feat = np.reshape(train_features, (-1, 1024))
    print('validation data: {}, {}'.format(train_feat.shape, train_label.size)) 
    np.save(open('features/' + case + '/train_features.npy', 'wb'), train_feat)
    np.save(open('features/' + case + '/train_labels.npy', 'wb'), train_label)

    val_gen = flow_generator_def(directory=val_dir, image_size=image_size, batch_size=batch_size, mode=mode)
    val_features = model.predict_generator(val_gen, val_gen.samples / batch_size)
    val_label = val_gen.classes
    val_feat = np.reshape(val_features, (-1, 1024))
    print('validation data: {}, {}'.format(val_feat.shape, val_label.size))    
    np.save(open('features/' + case + '/val_features.npy', 'wb'), val_feat)
    np.save(open('features/' + case + '/val_labels.npy', 'wb'), val_label)

if __name__ == "__main__":
    main()
