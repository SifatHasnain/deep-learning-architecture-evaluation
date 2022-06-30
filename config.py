import os
from unittest import result

data = "DogvCat"
# data = "mnist"

if data == "DogvCat":
    num_classes = 2
    class_names = ['cat', 'dog']
elif data =="mnist":
    num_classes= 10
    class_names= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

image_height = 227
image_width = 227
num_channels = 3

EPOCHS = 10
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
    
optimizer_fn = "sgd"
# optimizer_fn = "adam"
# optimizer_fn = "rmsprop"

learning_rate = 0.01
momentum = 0.9

# networks
model = "cnn"
#model = "alexnet"
# model = "resnet"
# model = "inceptionv1"
# model = "inceptionv2"
# model = "inceptionv3"
# model = "inceptionv4"

# paths and directories
result_dir = '/content/deep-learning-model-evaluation/results'
train_dir = "/content/deep-learning-model-evaluation/datasets/"+data+"/train"
valid_dir = "/content/deep-learning-model-evaluation/datasets/"+data+"/valid"
test_dir = "/content/deep-learning-model-evaluation/datasets/"+data+"/test"


# model version
## follow this versioning rule 
### cnn -> 1.0.
### alexnet -> 2.0.
### resnet -> 3.0.
### inception -> 4.0.
version = '1.0'
