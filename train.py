from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau

import config
import models
from dataprep import generate_dataset

def get_model():
    model = models.Alexnet()
    # if config.model == "resnet18":
    #     model = resnet_18()
    model.build(input_shape=(config.image_height, config.image_width, config.channels))
    model.summary()
    return model

if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

# get the original_dataset
    train_dataset, valid_dataset, test_dataset = generate_dataset()

# create model
    model = get_model()
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)

    callback_list = [rlrop, tensorboard_callback]

    # start training
    model.fit(train_dataset,
                        epochs= config.EPOCHS,
                        steps_per_epoch= train_dataset.samples // config.BATCH_SIZE,
                        validation_data=valid_dataset,
                        validation_steps= valid_dataset.samples // config.BATCH_SIZE,
                        callbacks=callback_list,
                        verbose=0)
    model.save(config.model_dir)
