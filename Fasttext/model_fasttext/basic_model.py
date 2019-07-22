'''
@Description: This is the basis model of the 
@Author: your name
@Date: 2019-07-13 22:55:34
@LastEditTime: 2019-07-22 16:54:44
@LastEditors: Please set LastEditors
'''
import tensorflow as tf 
from tensorflow.python import debug as tf_debug
import abc
from functools import reduce
import datetime
import numpy as np
class Model(object):
    """
    this is the basis model
        :param object: 
    """
    def __init__(self,opt):
        # initialize the parameters
        for key, value in opt.items():
            self.__setattr__(key, value)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.para = []
        self.build_graph()
        # summary
        self.merged = tf.summary.merge_all()
        # self.train_writer = tf.summary.FileWriter(self.summaries_dir + '/train',
        #                                           self.sess.graph)
        # self.test_writer = tf.summary.FileWriter(self.summaries_dir + '/test')
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        # whether debug the code
        # if self.debug:
        #     self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
    def _create_placeholder(self):
        self.sentence = tf.placeholder(tf.int32,[None,self.max_input_sentence],name = 'input_question')
        self.input_y = tf.placeholder(tf.float32, [None,2], name = "input_y")
        self.sentence_position = tf.placeholder(tf.int32,[None,self.max_input_sentence],name = 'q_position')
        
    @abc.abstractmethod
    def _get_embedding(self):
        """
        abstract method
            :param self: 
        """
    def _feed_neural_work(self):
        with tf.name_scope('regression'):
            regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
            W = tf.get_variable(
                "W_output",
                shape = [self.embedding_dim, 2],
                initializer = tf.contrib.layers.xavier_initializer(),
                regularizer=regularizer)
            b = tf.get_variable('b_output', shape=[2],initializer = tf.random_normal_initializer(),regularizer = regularizer)
            self.para.append(W)
            self.para.append(b)
            self.logits = tf.nn.xw_plus_b(self.represent, W, b, name = "scores")
            self.scores = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.scores, 1, name = "predictions")

    def _create_loss(self):
        l2_loss = tf.constant(0.0)
        for p in self.para:
            l2_loss += tf.nn.l2_loss(p)
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda*l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def _create_op(self):
    
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(
            self.grads_and_vars, global_step=self.global_step)

    def variable_summaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    @staticmethod
    def _model_stats():
        """Print trainable variables and total model size."""

        def size(v):
            return reduce(lambda x, y: x * y, v.get_shape().as_list())
        print("Trainable variables")
        for v in tf.trainable_variables():
            print("  %s, %s, %s, %s" %
                  (v.name, v.device, str(v.get_shape()), size(v)))
        print("Total model size: %d" % (sum(size(v)
                                            for v in tf.trainable_variables())))


    def _train(self,data_batch,i):

        for data in data_batch:
            sentence,flag,position = zip(*data)
            feed_dict = {
                self.sentence: sentence,
                self.input_y: flag,
                self.sentence_position: position
            }
            _, step, loss, accuracy = self.sess.run(
                [self.train_op, self.global_step, self.loss, self.accuracy],feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}  ".format( time_str, step, loss, accuracy))
    def _predict(self,data_batch):
        scores = []
        for data in data_batch:
            sentence,_,position = zip(*data)
            feed_dict = {
                self.sentence: sentence,
                self.sentence_position: position
            }
            score = self.sess.run(self.scores, feed_dict)
            scores.extend(score)
        return np.array(scores)

    def build_graph(self):
        self._create_placeholder()
        self._get_embedding()
        self._feed_neural_work()
        self._create_loss()
        self._create_op()



