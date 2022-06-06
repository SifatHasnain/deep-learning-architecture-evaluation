import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generate_train_dataset(config):
    #train
    train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.1)

    train_generator = train_datagen.flow_from_directory(config[0],
                                                        target_size=(config[2], config[3]),
                                                        color_mode="rgb",
                                                        batch_size=config[4],
                                                        seed=1,
                                                        shuffle=True,
                                                        class_mode="categorical")

    #valid
    valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
    valid_generator = valid_datagen.flow_from_directory(config[1],
                                                        target_size=(config[2], config[3]),
                                                        color_mode="rgb",
                                                        batch_size=config[5],
                                                        seed=7,
                                                        shuffle=True,
                                                        class_mode="categorical"
                                                        )
    
    return train_generator, valid_generator

def generate_test_dataset(config):
    #test
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(config[0],
                                                        target_size=(config[1], config[2]),
                                                        color_mode="rgb",
                                                        batch_size=config[3],
                                                        seed=7,
                                                        shuffle=True,
                                                        class_mode="categorical"
                                                        )
    return test_generator
