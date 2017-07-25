import nltk
import os
import numpy as np
import tqdm
import random
import pickle
from sklearn.model_selection import train_test_split

import tensorflow as tf

# Hyper-Parameter
window_size = 5
feature_size = 100
hidden_unit = 500
learning_rate = 1e-4

# data = []
# file_list = os.listdir("aclImdb/train/pos")
# for filename in tqdm.tqdm(file_list):
#     with open(os.path.join("aclImdb/train/pos", filename), "rt", encoding="utf-8") as f:
#         doc = f.read()
#         data.append(doc)
#
#
# def n_gram(words: list, n=window_size):
#     return [(words[i: i +n], words[i + n]) for i in range(len(words) - n)]
#
# vocab = set()
# dataset = []
#
# for doc in tqdm.tqdm(data[:100]):
#     words = nltk.word_tokenize(doc)
#     dataset += n_gram(words)
#     vocab.update(words)
#
#
# with open("dataset.pkl", "wb") as f:
#     f.write(pickle.dumps(dataset))
#
# with open("vocab.pkl", "wb") as f:
#     f.write(pickle.dumps(vocab))

with open("dataset.pkl", "rb") as f:
    dataset = pickle.loads(f.read())
with open("vocab.pkl", "rb") as f:
    vocab = pickle.loads(f.read())

vocab = list(vocab)
print("DataSet Size: ", len(dataset))
print("Vocabulary Size: ", len(vocab))


def one_hot(d):
    if type(d) == str:
        index = vocab.index(d)
        r = np.zeros((len(vocab), 1))
        r[index] = 1
        return r

    if type(d) == list:
        r = []
        for w in d:
            index = vocab.index(w)
            zeros = np.zeros((len(vocab)))
            zeros[index] = 1
            r.append(zeros)
        return np.array(r)


train, test = train_test_split(dataset, train_size=0.7, test_size=0.3, random_state=100)
train, valid = train_test_split(train, train_size=0.9, test_size=0.1, random_state=random.randint(0, 100))

X = tf.placeholder(tf.float32, (None, len(vocab)))
Y = tf.placeholder(tf.float32, (len(vocab), 1))

with tf.name_scope("projection"):
    proj_w = tf.get_variable("proj_w", shape=(len(vocab), feature_size), initializer=tf.contrib.layers.xavier_initializer())
    proj_b = tf.get_variable("proj_b", shape=(feature_size), initializer=tf.zeros_initializer())

    projection_layer = tf.add(tf.matmul(X, proj_w), proj_b)

    tf.summary.histogram("proj_w", proj_w)
    tf.summary.histogram("proj_b", proj_b)
    tf.summary.histogram("proj", projection_layer)

input_x = tf.reshape(projection_layer, (-1, 1))

hidden_w = tf.get_variable("hidden_w", shape=(hidden_unit, window_size * feature_size), initializer=tf.contrib.layers.xavier_initializer())
hidden_b = tf.get_variable("hidden_b", shape=(window_size * feature_size))
hidden_layer = tf.tanh(tf.add(tf.matmul(hidden_w, input_x), hidden_b))

U = tf.get_variable("U", shape=(len(vocab), hidden_unit))
output = tf.matmul(U, hidden_layer)

output_w = tf.get_variable("output_w", shape=(window_size * feature_size, 1), initializer=tf.contrib.layers.xavier_initializer())
output_b = tf.get_variable("output_b", shape=(1), initializer=tf.zeros_initializer())

output = tf.add(tf.matmul(output, output_w), output_b)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output, dim=0))
tf.summary.scalar("cost", cost)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost)

merged = tf.summary.merge_all()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

train_writer = tf.summary.FileWriter("./logs/train", graph=sess.graph)
saver = tf.train.Saver()
global_step = 0

for epoch in range(10):
    print("{} Epoch Start".format(epoch))
    for x, y in tqdm.tqdm(train):
        _ = sess.run(train_op, feed_dict={X: one_hot(x), Y: one_hot(y)})
        global_step += 1
        if global_step % 500 == 0:
            c, summary = sess.run([cost, merged], feed_dict={X: one_hot(x), Y: one_hot(y)})
            print("Cost: ", c)
            train_writer.add_summary(summary, global_step)
            saver.save(sess, "./models/model_{}.ckpt".format(global_step))
