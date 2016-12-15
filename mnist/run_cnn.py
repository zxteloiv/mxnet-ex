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

data = mx.symbol.Variable('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

import logging
logging.getLogger().setLevel(logging.DEBUG)

model = mx.model.FeedForward(
        ctx = mx.gpu(0),     # use GPU 0 for training, others are same as before
        symbol = lenet,       
        num_epoch = 10,     
        learning_rate = 0.1)

model.fit(
        X=train_iter,  
        eval_data=val_iter, 
        batch_end_callback = mx.callback.Speedometer(batch_size, 200)
        ) 

prob = model.predict(to4d(val_img[0:1]))[0]
print 'Classified as %d with probability %f' % (prob.argmax(), max(prob))

print 'Validation accuracy: %f%%' % (model.score(val_iter)*100,)
