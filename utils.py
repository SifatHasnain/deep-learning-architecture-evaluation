import os
from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow import keras

import config
from dataprep import generate_test_dataset 

if __name__ == '__main__':
    
    # get the test_dataset
    test_data_config = (config.test_dir, config.image_height, config.image_width, config.TEST_BATCH_SIZE)
    test_dataset = generate_test_dataset(test_data_config)
        
    x_test, label_batch  = next(iter(test_dataset))

    result_save_path = os.path.join(config.result_dir, config.model)
    model_name="{}_{}_dogcat.h5".format(config.model, config.version)
    model_save_path = os.path.join(result_save_path, model_name)
    
    model=keras.models.load_model(model_save_path)

    # plot the images: each image is 227x227 pixels
    # for i in range(8):
    #     ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
    #     ax.imshow(x_test[i,:],cmap=plt.cm.gray_r, interpolation='nearest')
    
    #     if prediction_values[i] == np.argmax(label_batch[i]):
    #         # label the image with the blue text
    #         ax.text(3, 17, config.class_names[prediction_values[i]], color='blue', fontsize=14)
    #     else:
    #         # label the image with the red text
    #         ax.text(3, 17, config.class_names[prediction_values[i]], color='red', fontsize=14)

    #confusion matrix
    predict_x=model.predict(x_test) 

    pred = np.round(predict_x)
    pred = np.argmax(pred,axis=1)

    label = np.argmax(label_batch,axis=1)

    pred = pred.tolist()
    label = label.tolist()

    cf= confusion_matrix(pred,label)
    
    accuracy = model.evaluate(x_test,label_batch)

    #return pred, label, x_test, cf, accuracy
