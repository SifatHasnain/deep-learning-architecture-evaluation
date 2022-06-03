import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import image_height, image_width, train_dir,valid_dir,test_dir, BATCH_SIZE

def generate_dataset():
    #train
    train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.1)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(image_height, image_width),
                                                        color_mode="rgb",
                                                        batch_size=BATCH_SIZE,
                                                        seed=1,
                                                        shuffle=True,
                                                        class_mode="categorical")
    train_num = train_generator.samples

    #valid
    valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
    valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                        target_size=(image_height, image_width),
                                                        color_mode="rgb",
                                                        batch_size=BATCH_SIZE,
                                                        seed=7,
                                                        shuffle=True,
                                                        class_mode="categorical"
                                                        )
    valid_num = valid_generator.samples

    #test
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(test_dir,
                                                        target_size=(image_height, image_width),
                                                        color_mode="rgb",
                                                        batch_size=BATCH_SIZE,
                                                        seed=7,
                                                        shuffle=True,
                                                        class_mode="categorical"
                                                        )
    test_num = test_generator.samples
    
    return train_generator, valid_generator, test_generator