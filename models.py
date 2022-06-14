import re
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Add,Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, AveragePooling2D,ZeroPadding2D
from tensorflow.keras import layers

# AlexNet model
class AlexNet(Model):
    def __init__(self, input_shape, num_classes):
        super(AlexNet, self).__init__(name='')

        self.conv1 = Conv2D(96, kernel_size=(11,11), strides= 4,
                        padding= 'valid', activation= 'relu',
                        input_shape= input_shape,
                        kernel_initializer= 'he_normal')
                        
        self.maxpool1 = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None)

        self.conv2 = Conv2D(256, kernel_size=(5,5), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal')
        self.maxpool2 = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None)

        self.conv3 = Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal')

        self.conv4 = Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal')

        self.conv5 = Conv2D(256, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal')

        self.maxpool3 = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None)
    
        self.flatten1 = Flatten()
        self.dropout1 = Dropout(0.5)
        self.fc1 = Dense(2048, activation= 'relu')
        self.dropout2 = Dropout(0.5)
        self.fc2 = Dense(2048, activation= 'relu')
        self.fc3 = Dense(1000, activation= 'relu')
        self.fc4 = Dense(num_classes, activation= 'softmax')
        
    def call(self, input_tensor, training=False):
        
        x = self.conv1(input_tensor)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.maxpool2(x)
    
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool3(x)
        x = self.flatten1(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
    
        return x
#cnn
class Conv(layers.Layer):
  def __init__(self, filters, kernel_size):
        super(Conv, self).__init__(name='')

        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        
  def call(self, inputs, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.pool(x)
        return x
class CNN(Model):
  def __init__(self, input_shape, num_classes):
    super(CNN, self).__init__(name='')
    self.conv1 = Conv(filters=32, kernel_size=(3, 3))
    self.conv2 = Conv(filters=64, kernel_size=(3, 3))
    self.conv3 = Conv(filters=128, kernel_size=(3, 3))
    self.conv4 = Conv(filters=128, kernel_size=(3, 3))
    self.flatten = Flatten()
    self.fc1 = Dense(512, activation= 'relu')
    self.fc2 = Dense(2, activation= 'relu')
  
  def call(self, input_tensor, training=False):
    x = self.conv1(input_tensor)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.fc2(x)

    return x

class Iden(layers.Layer):
  def __init__(self, X, f, filters, stage, block,s):
        super(Iden, self).__init__(name='')
        
        # defining base name for block
        conv_base_name = 'res' + str(stage) + block + '_'
        bn_base_name = 'bn' + str(stage) + block + '_'

        # retrieve number of filters in each layer of main path
        # NOTE: f3 must be equal to n_C. That way dimensions of the third component will match the dimension of original input to identity block
        f1, f2, f3 = filters

        # Batch normalization must be performed on the 'channels' axis for input. It is 3, for our case
        bn_axis = 3
        
        skip_conn1 = X
        
        #first component-main path
        self.conv = Conv2D(f1, (1, 1), strides = (s,s), padding = 'valid', name = conv_base_name + 'first_component', kernel_initializer = glorot_uniform(seed=0))
        self.bn = BatchNormalization(axis = bn_axis, name = bn_base_name + 'first_component')
        self.act = Activation('relu')
        
        #second component- main path
        self.conv2 = Conv2D(f2,  kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_base_name + 'second_component', kernel_initializer = glorot_uniform(seed=0))
        self.bn2 = BatchNormalization(axis = bn_axis, name = bn_base_name + 'second_component')
        self.act2 = Activation('relu')
        
        #third component-  main path
        self.conv3 = Conv2D(f3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_base_name + 'third_component', kernel_initializer = glorot_uniform(seed=0))
        self.bn3 = BatchNormalization(axis = bn_axis, name = bn_base_name + 'third_component')
        
        # #skip connection
        # self.skip_con1 = Conv2D(f3, (1, 1), strides = (s,s), padding = 'valid', name = conv_base_name + 'merge', kernel_initializer = glorot_uniform(seed=0))
        # self.skip_conn2 = BatchNormalization(axis = 3, name = bn_base_name + 'merge')
        
        # "Addition step" 
    #both values have same dimensions at this point
        self.add = Add()([X, skip_conn1])
        self.act3 = Activation('relu')
    
  def call(self, inputs):
      x= self.conv(x)
      x= self.bn(x)
      x= self.act(x)
      
      x= self.conv2(x)
      x=self.bn2(x)
      x=self.act2(x)
      
      x=self.conv3(x)
      x=self.bn3(x)
      
      x=self.add(x)
      x=self.act3(x)
      return x
      

class Conv(layers.Layer):
  def __init__(self, X, f, filters, stage, block,s):
        super(Conv, self).__init__(name='')
        
        # defining base name for block
        conv_base_name = 'res' + str(stage) + block + '_'
        bn_base_name = 'bn' + str(stage) + block + '_'

        # retrieve number of filters in each layer of main path
        # NOTE: f3 must be equal to n_C. That way dimensions of the third component will match the dimension of original input to identity block
        f1, f2, f3 = filters

        # Batch normalization must be performed on the 'channels' axis for input. It is 3, for our case
        bn_axis = 3
        
        # save input for "addition" to last layer output; step in skip-connection
        skip_conn1 = X
    
        #first component-main path
        self.conv = Conv2D(f1, (1, 1), strides = (s,s), padding = 'valid', name = conv_base_name + 'first_component', kernel_initializer = glorot_uniform(seed=0))
        self.bn = BatchNormalization(axis = bn_axis, name = bn_base_name + 'first_component')
        self.act = Activation('relu')
        
        #second component- main path
        self.conv2 = Conv2D(f2,  kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_base_name + 'second_component', kernel_initializer = glorot_uniform(seed=0))
        self.bn2 = BatchNormalization(axis = bn_axis, name = bn_base_name + 'second_component')
        self.act2 = Activation('relu')
        
        #third component-  main path
        self.conv3 = Conv2D(f3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_base_name + 'third_component', kernel_initializer = glorot_uniform(seed=0))
        self.bn3 = BatchNormalization(axis = bn_axis, name = bn_base_name + 'third_component')
        
        #skip connection
        skip_conn1 = Conv2D(f3, (1, 1), strides = (s,s), padding = 'valid', name = conv_base_name + 'merge', kernel_initializer = glorot_uniform(seed=0))(skip_conn1)
        skip_conn2 = BatchNormalization(axis = 3, name = bn_base_name + 'merge')(skip_conn1)
        
        # "Addition step" 
    #both values have same dimensions at this point
        self.add = Add()([X, skip_conn2])
        self.act3 = Activation('relu')
    
  def call(self, inputs):
      x= self.conv(x)
      x= self.bn(x)
      x= self.act(x)
      
      x= self.conv2(x)
      x=self.bn2(x)
      x=self.act2(x)
      
      x=self.conv3(x)
      x=self.bn3(x)
      
      x=self.add(x)
      x=self.act3(x)
      return x
  

class Resnet(Model):
    def __init__(self, input_shape, num_classes):
        super(Resnet, self).__init__(name='')
        self.input_shapes= input_shape
        self.num_classes= num_classes

    def call(self, input_tensor, training=False):
        # plug in input_shape to define the input tensor
        X_input = Input(self.input_shapes)

        # Zero-Padding : pads the input with a pad of (3,3)
        X = ZeroPadding2D((3, 3))(X_input)
        
        # Stage 1
        X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv_0', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)
   
        # Stage 2
        X = Conv(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
        X = Iden(X, 3, [64, 64, 256], stage=2, block='b')
        X = Iden(X, 3, [64, 64, 256], stage=2, block='c')

        # Stage 3
        X = Conv(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
        X = Iden(X, 3, [128, 128, 512], stage=3, block='b')
        X = Iden(X, 3, [128, 128, 512], stage=3, block='c')
        X = Iden(X, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4
        X = Conv(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
        X = Iden(X, 3, [256, 256, 1024], stage=4, block='b')
        X = Iden(X, 3, [256, 256, 1024], stage=4, block='c')
        X = Iden(X, 3, [256, 256, 1024], stage=4, block='d')
        X = Iden(X, 3, [256, 256, 1024], stage=4, block='e')
        X = Iden(X, 3, [256, 256, 1024], stage=4, block='f')

        # Stage 5
        X = Conv(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
        X = Iden(X, 3, [512, 512, 2048], stage=5, block='b')
        X = Iden(X, 3, [512, 512, 2048], stage=5, block='c')

        # Average Pooling
        X = AveragePooling2D((2, 2), name='avg_pool')(X)

        # output layer
        X = Flatten()(X)
        X = Dense(self.num_classes, activation='softmax', name='fc' + str(self.num_classes), kernel_initializer = glorot_uniform(seed=0))(X)
            
        return X    
        

