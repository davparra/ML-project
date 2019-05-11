from keras.preprocessing.image import ImageDataGenerator

def flow_generator(directory, image_size, batch_size, mode):
    datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_directory(
        directory=directory,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode=mode)

    return generator

def flow_generator_def(directory, image_size, batch_size):
    datagen = ImageDataGenerator(
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255)

    generator = datagen.flow_from_directory(
        directory=directory,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode=None)

    return generator