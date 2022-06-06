import os
from unittest import result

num_classes = 2
class_names = ['cat', 'dog']

image_height = 227
image_width = 227
num_channels = 3

EPOCHS = 30
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
TEST_BATCH_SIZE = 32

optimizer = "sgd"
optimizer = "adam"
optimizer = "rmsprop"

learning_rate = 0.01
momentum = 0.9

# networks
# model = "cnn"
model = "alexNet"
# model = "resnet"
# model = "inception"

# paths and directories
result_dir = 'results'
train_dir = "./datasets/dogcat/train"
valid_dir = "./datasets/dogcat/validation"
test_dir = "./datasets/dogcat/test"

# model version
## follow this versioning rule 
### cnn -> 1.0.
### alexnet -> 2.0.
### resnet -> 3.0.
### inception -> 4.0.
version = '1.0'