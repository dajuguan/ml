import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt

batch_size = 100
vocabulary_size = 4000 + 3
num_nodes = 252
epochs = 1000


class BatchGenerator(object):
    def __init__(self, batch_size, text_list, labels, epoch=1, current_place=0):
        self.batch_size = batch_size
        self.epoch = epoch
        self.text_list = text_list
        self.labels = labels
        self.current_place = current_place

    def next(self):
        if self.current_place + self.batch_size > len(self.text_list):
            self.epoch += 1
            self.current_place = 0
        text = tf.cast(np.array(self.text_list[self.current_place: self.current_place + self.batch_size]),
                       dtype=tf.int32)
        y = tf.one_hot(self.labels[self.current_place: self.current_place + self.batch_size], depth=2, dtype=tf.float32)
        x = tf.one_hot(text, depth=4003, dtype=np.float32)
        self.current_place += self.batch_size
        return x, y, self.epoch


h = h5py.File('cleaned_with_top_4k_used/data.h5')
labeled_train = np.array(h['labeledTrainData'])
import pandas as pd

labels = pd.read_csv('sentiment/labeledTrainData.tsv', sep='\t', header=0,
                     error_bad_lines=False, encoding='ISO-8859-1')['sentiment'].values
unlabeled_train = np.array(h['unlabeledTrainData'])

test_data = np.array(h['testData'])
bg_train = BatchGenerator(batch_size, labeled_train, labels)
# bg_test = BatchGenerator(100, test_data)

xs = tf.placeholder(dtype=tf.float32, shape=[None, num_nodes, vocabulary_size])
ys = tf.placeholder(dtype=tf.float32, shape=[None, 2])
rnn_cell = tf.nn.rnn_cell.LSTMCell(num_nodes)

_, state = tf.nn.dynamic_rnn(cell=rnn_cell,
                             inputs=xs,
                             dtype=tf.float32,
                             )

outputs3 = tf.layers.flatten(tf.concat(state,axis=1))
print(outputs3)
outputs4 = tf.layers.Dense(units=250, activation='relu')(outputs3)
print(outputs4)
outputs5 = tf.layers.Dense(2, activation='sigmoid')(outputs4)
print(outputs5)
loss = tf.losses.sigmoid_cross_entropy(ys,outputs5)
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
Loss = []
while (True):
    x_batch, y_batch, epoch = bg_train.next()
    if epoch > epochs:
        break
    _, err = sess.run([train, loss], feed_dict={xs: sess.run(x_batch), ys: sess.run(y_batch)})
    print(err)
    Loss.append(err)
plt.plot(Loss)
