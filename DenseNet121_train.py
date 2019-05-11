from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, EarlyStopping
from DenseNet121 import load_DenseNet121
from fine_tuning import freeze, freeze_all, unfreeze, unfreeze_all
from dir import get_train_case_dir, get_val_case_dir
from data import plot_data, store_accuracies, store_loss, print_layers
from dataflow import flow_generator
from pprint import pprint
from time import time

#gpu settings
#tf.ConfigProto().gpu_options.per_process_gpu_memory_fraction = 0.7

#tensorboard logs
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

#Settings
case = '2'
image_size = 160
batch_size = 8
epochs = 40
optimizer = Adam(lr=1e-4)
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

train_dir = get_train_case_dir(case)
val_dir = get_val_case_dir(case)

training_flow = flow_generator(directory=train_dir, 
                image_size=image_size,
                batch_size=batch_size,
                mode=mode)

validation_flow = flow_generator(directory=val_dir, 
                image_size=image_size,
                batch_size=batch_size,
                mode=mode)

print(validation_flow.class_indices)

model = load_DenseNet121(input_shape=IMG_SHAPE, classes=classes, optimizer=optimizer, function=function, verbose=0, trainable=True)

#training settings
steps_per_epoch = training_flow.n // batch_size
validation_steps = validation_flow.n // batch_size

# Save the model according to the conditions  
checkpoint_path = 'models/DenseNet121-' + case + '.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

history = model.fit_generator(generator=training_flow, steps_per_epoch=steps_per_epoch, epochs=epochs, workers=4, validation_data=validation_flow, validation_steps=validation_steps, verbose=1, callbacks=[checkpoint, early, tensorboard])

 # accuracy and loss graph
accuracies = history.history['acc']
val_accuracies = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

#TODO: add fine tunning
print('Fine Tuning Starting...')
#unfreezing bottom layers
#freeze_all(model)
#unfreeze(model, fine_tune_at)
unfreeze_all(model)
freeze(model, fine_tune_at)

#update learn rate of optimizer and recompile
optimizer = Adam(lr=2e-6)

model.compile(optimizer=optimizer,
            loss=function,
            metrics=['accuracy'])

history_ft = model.fit_generator(generator=training_flow, steps_per_epoch=steps_per_epoch, epochs=epochs, workers=4, validation_data=validation_flow, validation_steps=validation_steps, verbose=1, callbacks=[checkpoint, early, tensorboard])

# learning curves
accuracies += history_ft.history['acc']
val_accuracies += history_ft.history['val_acc']

loss += history_ft.history['loss']
val_loss += history_ft.history['val_loss']

#store results
plot_data(accuracies, val_accuracies, loss, val_loss, epochs, case, net, function)
store_accuracies(history.history['val_acc'], case, net, function)
store_loss(history.history['val_loss'], case, net, function)

model.save('models/case' + case + '_' + net +'.h5')