import mnist_data

train_img = mnist_data.train_img
train_lbl = mnist_data.train_lbl
val_img = mnist_data.val_img
val_lbl = mnist_data.val_lbl

import mxnet as mx
import numpy as np

def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

batch_size = 100
train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)

# Create a place holder variable for the input data
data = mx.sym.Variable('data')
# Flatten the data from 4-D shape (batch_size, num_channel, width, height) 
# into 2-D (batch_size, num_channel*width*height)
data = mx.sym.Flatten(data=data)

# The first fully-connected layer
fc1  = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
# Apply relu to the output of the first fully-connnected layer
act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")

# The second fully-connected layer and the according activation function
fc2  = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden = 64)
act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")

# The thrid fully-connected layer, note that the hidden size should be 10, which is the number of unique digits
fc3  = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)
# The softmax and loss layer
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

# We visualize the network structure with output size (the batch_size is ignored.)
#shape = {"data" : (batch_size, 1, 28, 28)}
#mx.viz.plot_network(symbol=mlp, shape=shape)

import logging
logging.getLogger().setLevel(logging.DEBUG)

model = mx.model.FeedForward(
        ctx=mx.gpu(0),
        symbol=mlp,       # network structure
        num_epoch=10,     # number of data passes for training 
        learning_rate=0.1 # learning rate of SGD 
        )

model.fit(
        X=train_iter,       # training data
        eval_data=val_iter, # validation data
        batch_end_callback = mx.callback.Speedometer(batch_size, 200) # output progress for each 200 data batches
        ) 

prob = model.predict(val_img[0:1].astype(np.float32)/255)[0]
print 'Classified as %d with probability %f' % (prob.argmax(), max(prob))

print 'Validation accuracy: %f%%' % (model.score(val_iter)*100,)
