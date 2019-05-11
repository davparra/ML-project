from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, AveragePooling2D
from keras.regularizers import l2
from keras import Sequential
from data import print_layers

def load_DenseNet121(input_shape, classes, optimizer, function, include_top=True, verbose=0, compile=False, trainable=False):
    #loading DenseNet121 pretrained model 
    base_model = DenseNet121(input_shape=input_shape, include_top=False, weights='imagenet')

    #freeze layers if trainable = false
    for layer in base_model.layers:
            layer.trainable = trainable

    if include_top:
        if classes == 2:
            #binary case
            x = base_model.output
            x = Dense(128)(x)
            x = GlobalAveragePooling2D(name='gavp')(x)
            predictions = Dense(classes, activation='softmax', name='soft')(x)

            model = Model(inputs=base_model.input, output=predictions)

            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            #adding a global spatial average pooling layer
            x = base_model.output
            x = Flatten(input_shape=input_shape, name='f')(x)
            x = Dense(1024, activation='relu', name='d1', kernel_regularizer=l2(.01))(x)  
            x = Dropout(0.5)(x)
            x = Dense(1024, activation="relu")(x) 
            #logistic layer for prediction
            predictions = Dense(classes, activation='softmax', name='d2')(x)

            model = Model(inputs=base_model.input, output=predictions)

            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        x = base_model.output
        out = AveragePooling2D((7, 7), name='avg_pool_app')(x)
        model = Model(inputs=base_model.input, output=out)

        if classes == 2:
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        

    if verbose == 1:
        model.summary()
    elif verbose == 2:
        print_layers(model)

    return model