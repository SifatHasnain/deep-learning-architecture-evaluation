from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2
import urllib
import requests
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import os

from train import model
from config import class_names
from dataprep import generate_dataset 

# get the original_dataset
train_dataset, valid_dataset, test_dataset = generate_dataset()
    
x_test, label_batch  = next(iter(test_dataset))

prediction_values = model.predict_classes(x_test)
# set up the figure
fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the images: each image is 227x227 pixels
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(x_test[i,:],cmap=plt.cm.gray_r, interpolation='nearest')
  
    if prediction_values[i] == np.argmax(label_batch[i]):
        # label the image with the blue text
        ax.text(3, 17, class_names[prediction_values[i]], color='blue', fontsize=14)
    else:
        # label the image with the red text
        ax.text(3, 17, class_names[prediction_values[i]], color='red', fontsize=14)
#confusion matrix
predict_x=model.predict(x_test) 

pred = np.round(predict_x)
pred = np.argmax(pred,axis=1)

label = np.argmax(label_batch,axis=1)

pred = pred.tolist()
label = label.tolist()

cf= confusion_matrix(pred,label)
print(cf)

plt.figure(figsize=(5,4))
sns.heatmap(cf, annot=True, xticklabels=['cat','dog'], yticklabels=['cat','dog'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()
model.evaluate(x_test,label_batch)