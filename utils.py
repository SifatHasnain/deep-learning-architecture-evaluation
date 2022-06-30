import os

import numpy as np

from config import *

from dataprep import generate_test_dataset 

from tensorflow import keras
from sklearn.metrics import confusion_matrix
import config

def evaluate():
    # get the test_dataset
    test_data_config = (test_dir, image_height, image_width, TEST_BATCH_SIZE)
    test_dataset = generate_test_dataset(test_data_config)
        

    result_save_path = os.path.join(result_dir, model)
    
    if config.data == "DogvCat":
        model_name="{}_{}_dogcat".format(config.model, config.version)
    elif config.data == "mnist":
        model_name="{}_{}_mnist".format(config.model, config.version)  
        
    # model_name = "{}_{}_dogcat".format(model, version)
    
    model_save_path = os.path.join(result_save_path, model_name)
   
    loaded_model = keras.models.load_model(model_save_path)
    
    #confusion matrix
    predict_x = loaded_model.predict(test_dataset) 
    
    pred = np.round(predict_x)
    pred = np.argmax(pred,axis=1)

    pred = pred.tolist()
    
    label = test_dataset.classes.tolist()
    
    cf= confusion_matrix(pred, label)
    
    accuracy = loaded_model.evaluate(test_dataset)

    return pred, test_dataset, cf, accuracy