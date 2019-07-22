'''
@Description: In User Settings Edit
@Author: suzhan
@Date: 2019-07-14 15:09:08
@LastEditTime: 2019-07-22 16:50:12
@LastEditors: Please set LastEditors
'''
from model_fasttext.basic_model import Model
import tensorflow as tf 
from multiply import ComplexMultiply
import numpy as np
class Fasttext(Model):
	def _get_embedding(self):
		with tf.name_scope("embedding"):
			self.embedding_W = tf.Variable(np.array(self.embeddings),name = "W" ,dtype="float32",trainable = self.trainable)
			self.embedding_W_pos =tf.Variable(self.Position_Embedding(self.embedding_dim),name = 'W',trainable = self.trainable)
			self.embedded_chars_q,self.embedding_chars_q_phase = self.concat_embedding(self.sentence,self.sentence_position)
			self.embedded_chars_q = tf.reduce_sum([self.embedded_chars_q,self.embedding_chars_q_phase],0)
			self.represent=tf.reduce_mean(self.embedded_chars_q,1)
			print(self.represent)
	def concat_embedding(self,words_indice,position_indice):
		embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice) #[batch,sen_len,embed_size]
		embedding_chars_q_phase=tf.nn.embedding_lookup(self.embedding_W_pos,words_indice)#[]batch,sen_len,embed_position_size
		pos=tf.expand_dims(position_indice,2)
		pos=tf.cast(pos,tf.float32)
		embedding_chars_q_phase=tf.multiply(pos,embedding_chars_q_phase)
		[embedded_chars_q, embedding_chars_q_phase] = ComplexMultiply()([embedding_chars_q_phase,embedded_chars_q])
		return embedded_chars_q,embedding_chars_q_phase
	def Position_Embedding(self,position_size):
		seq_len = self.vocab_size
		position_j = 1. / tf.pow(10000., 2 * tf.range(position_size, dtype=tf.float32) / position_size)
		position_j = tf.expand_dims(position_j, 0)

		position_i=tf.range(tf.cast(seq_len,tf.float32), dtype=tf.float32)
		position_i=tf.expand_dims(position_i,1)
		position_ij = tf.matmul(position_i, position_j)
		position_embedding = position_ij
		return position_embedding