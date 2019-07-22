'''
@Description: begin
@Author: zhansu
@Date: 2019-07-15 17:01:25
@LastEditTime: 2019-07-22 17:14:50
@LastEditors: Please set LastEditors
'''
import time
import logging
import os
import helper
import config
import tensorflow as tf
from model_fasttext.fasttext import Fasttext
import os 
import evaluation
import sys
import numpy as np 
sys.path.append('.')

now = int(time.time())
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)
log_dir = 'wiki_log/' + timeDay
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
program = os.path.basename("sentence classfication")
logger = logging.getLogger(program)

# 获得配置
FLAGS = config.flags.FLAGS
# FLAGS._parse_flags()
opts = FLAGS.flag_values_dict()
for item in opts:
    logger.info('{} : {}'.format(item, opts[item]))
############## 数据载入 #################
logger.info('load data ...........')
train,dev,test = helper.load(opts['data_dir'])

max_sent_length = max(map(lambda x: len(x), train['question'].str.split()))
max_sent_length = 33
############## 数据预处理 ###############
alphabet, embeddings,embeddings_complex = helper.prepare(
        [train, test, dev],dim=FLAGS.embedding_dim)
############## 模型预测 #################

opts['embeddings'] = embeddings
opts['max_input_sentence'] = max_sent_length
opts['vocab_size'] = len(alphabet)

with tf.Graph().as_default():
    model = Fasttext(opts)
    model._model_stats()

    for i in range(opts['num_epochs']):
        data_gen = helper.batch_iter(train,opts['batch_size'],alphabet,True,sen_len=max_sent_length)
        model._train(data_gen,i)

        # 测试代码

        test_data = helper.batch_iter(test,opts['batch_size'],alphabet,False,sen_len = max_sent_length)
        test_score = model._predict(test_data)
        test['predicted_label'] = np.argmax(test_score,1)
        print(evaluation.evaluationBypandas_f1_acc(test))
