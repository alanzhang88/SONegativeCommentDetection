from tokenize_words import DataHandler
import tensorflow as tf
import numpy as np

filePath='labeled_news.csv'
# sequence_length=50
# n_class=17
# embedding_size=40
# num_filters=20
#
# data = DataHandler(filePath=filePath,sequence_length=sequence_length,n_class=n_class)
# vocab_size = data.vocab_size
#
# # PLACEHOLDERS
# X = tf.placeholder(tf.int32,shape=[None,sequence_length])
# y_true = tf.placeholder(tf.int32,shape=[None,n_class])
# hold_prob = tf.placeholder(tf.float32)
#
# # VARIABLES
#
# # Embedding Layers
# W_1 = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0))
# embedding = tf.expand_dims(tf.nn.embedding_lookup(W_1,X),-1)
#
# # Conv Layers
# filter_list = [2,3,4,5,6]
# pooled_output = []
# for i, filter_size in enumerate(filter_list):
#     filter_shape = [filter_size,embedding_size,1,num_filters]
#     W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1))
#     b = tf.Variable(tf.constant(0.1,shape=[num_filters]))
#     conv = tf.nn.conv2d(embedding,W,strides=[1,1,1,1],padding='VALID')
#     h = tf.nn.relu(tf.nn.bias_add(conv,b))
#     pooled = tf.nn.max_pool(h,ksize=[1,sequence_length - filter_size + 1, 1, 1],strides=[1,1,1,1],padding='VALID')
#     pooled_output.append(pooled)
#
# total_filter = num_filters * len(filter_list)
# h_pool = tf.concat(pooled_output,3)
# h_pool_flat = tf.reshape(h_pool,[-1,total_filter])
#
# def init_weights(shape):
#     init_random_dist = tf.truncated_normal(shape,stddev=0.1)
#     return tf.Variable(init_random_dist)
# # INIT BIAS
# def init_bias(shape):
#     init_bias_vals = tf.constant(0.1,shape=shape)
#     return tf.Variable(init_bias_vals)
# def normal_full_layer(input_layer,size):
#     input_size = int(input_layer.get_shape()[1])
#     W = init_weights([input_size,size])
#     b = init_bias([size])
#     return tf.matmul(input_layer,W) + b
#
# full_layer_one = tf.nn.relu(normal_full_layer(h_pool_flat,1024))
# full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
# y_pred = normal_full_layer(full_one_dropout,n_class)
#
# # LOSS FUNC
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred))
#
# # OPTIMIZER
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
# train = optimizer.minimize(cross_entropy)
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
# steps = 1000
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     for i in range(steps):
#         batch_x, batch_y = data.next_batch(50)
#         sess.run(train,feed_dict={X:batch_x,y_true:batch_y,hold_prob:0.8})
#         if i%100 == 0:
#             print("ON STEP: {}".format(i))
#             print("ACCURACY: ")
#             matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
#             acc = tf.reduce_mean(tf.cast(matches,tf.float32))
#             test_X, test_y = data.test_batch()
#             print(sess.run(acc,feed_dict={X:test_X,y_true:test_y,hold_prob:1.0}))
#             print("\n")
#
#     test_X, test_y = data.test_batch()
#     print(sess.run(tf.argmax(y_pred,1),feed_dict={X:test_X,y_true:test_y,hold_prob:1.0}))
#     print('\n')
#     print(np.argmax(test_y,axis=1))
#     saver.save(sess,"./model/cnn_news_classifier")


class newsClassifier:
    def __init__(self,filePath,sequence_length=50,embedding_size=40,num_filters=20,n_class=17,hold_prob_raw=0.8,filter_list=[2,3,4,5,6]):
        self.filePath = filePath
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        self.n_class = n_class
        self.hold_prob_raw = hold_prob_raw
        self.data = DataHandler(filePath,sequence_length,n_class)
        self.vocab_size = self.data.vocab_size
        self.X = tf.placeholder(tf.int32,shape=[None,sequence_length],name='X')
        self.y_true = tf.placeholder(tf.int32,shape=[None,n_class],name='y_true')
        self.hold_prob = tf.placeholder(tf.float32,name='hold_prob')
        embd_W = tf.Variable(tf.random_uniform([self.vocab_size,embedding_size],-1.0,1.0))
        embedding = tf.expand_dims(tf.nn.embedding_lookup(embd_W,self.X),-1)
        pooled_output = []
        for i, filter_size in enumerate(filter_list):
            filter_shape = [filter_size,embedding_size,1,num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1))
            b = tf.Variable(tf.constant(0.1,shape=[num_filters]))
            conv = tf.nn.conv2d(embedding,W,strides=[1,1,1,1],padding='VALID')
            h = tf.nn.relu(tf.nn.bias_add(conv,b))
            pooled = tf.nn.max_pool(h,ksize=[1,sequence_length - filter_size + 1, 1, 1],strides=[1,1,1,1],padding='VALID')
            pooled_output.append(pooled)
        total_filter = num_filters * len(filter_list)
        h_pool = tf.concat(pooled_output,3)
        h_pool_flat = tf.reshape(h_pool,[-1,total_filter])
        full_layer_one = tf.nn.relu(self.normal_full_layer(h_pool_flat,1024))
        full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=self.hold_prob)
        self.y_pred = self.normal_full_layer(full_one_dropout,n_class)

    def init_weights(self,shape):
        init_random_dist = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(init_random_dist)

    def init_bias(self,shape):
        init_bias_vals = tf.constant(0.1,shape=shape)
        return tf.Variable(init_bias_vals)

    def normal_full_layer(self,input_layer,size):
        input_size = int(input_layer.get_shape()[1])
        W = self.init_weights([input_size,size])
        b = self.init_bias([size])
        return tf.matmul(input_layer,W) + b

    def train(self,learning_rate=0.001,steps=1000, model_dir="./model/cnn_news_classifier"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_true,logits=self.y_pred))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(cross_entropy)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)

            for i in range(steps):
                batch_x, batch_y = self.data.next_batch(50)
                sess.run(train,feed_dict={self.X:batch_x,self.y_true:batch_y,self.hold_prob:0.8})
                if i%100 == 0:
                    print("ON STEP: {}".format(i))
                    print("ACCURACY: ")
                    matches = tf.equal(tf.argmax(self.y_pred,1),tf.argmax(self.y_true,1))
                    acc = tf.reduce_mean(tf.cast(matches,tf.float32))
                    test_X, test_y = self.data.test_batch()
                    print(sess.run(acc,feed_dict={self.X:test_X,self.y_true:test_y,self.hold_prob:1.0}))
                    print("\n")
            saver.save(sess,model_dir)

    def classify(self,input_str,model_dir="./model/cnn_news_classifier"):

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,model_dir)
            feed_X = self.data.tokenize_strs([input_str])
            res = sess.run(tf.argmax(self.y_pred,1),feed_dict={self.X:feed_X,self.hold_prob:1.0})
            print(res[0]+1)
            return res[0].item()+1

# t = newsClassifier(filePath=filePath)
# t.classify("Why the Nunes memo really isn't a partisan fight Because we are so polarized politically these days, there's a tendency to assume that every single issue that breaches our collective national consciousness must, at its root, be a fight between Democrats and Republicans. CNN")
