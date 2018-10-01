import numpy as np
import tensorflow as tf
from bilm.model import BidirectionalLanguageModel,dump_token_embeddings
from bilm.elmo import weight_layers
from config import Params


class HAN:
	def __init__(self, feature_type=None, elmo_type=None, pre_embed=None):
	
		if feature_type is None and elmo_type is None:
			exit(0)

		self.lr = Params.lr
		self.word_dim = Params.word_dim
	
		self.num_classes = Params.num_classes
		self.batch_size = Params.batch_size
		self.hidden_dim = Params.hidden_dim
		self.sen_len = Params.sen_max_len
		self.sen_num = Params.doc_max_sen

		self.batch_sen_num = self.batch_size*self.sen_num


		self.keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')
		self.input_word = tf.placeholder(dtype=tf.int32,shape=[None,self.sen_len],name='input_word')
		self.input_word_ELMO = tf.placeholder(dtype=tf.int32,shape=[None,None],name='input_word_ELMO')
		self.input_label = tf.placeholder(dtype=tf.float32,shape=[None,self.num_classes],name='input_label')

		if feature_type in ['word', 'char']:

			self.word_embedding = tf.get_variable(initializer=pre_embed,name='word_embedding')
			all_input_words = tf.nn.embedding_lookup(self.word_embedding,self.input_word)
			all_input_words = tf.nn.dropout(all_input_words,self.keep_prob)

			layer1_forward = self.LSTM()
			layer1_backward = self.LSTM()
			with tf.variable_scope('LSTM'):
				all_output_words, _ = tf.nn.bidirectional_dynamic_rnn(layer1_forward,layer1_backward,all_input_words,dtype=tf.float32)
			all_output_words = tf.concat(axis=2,values=all_output_words)

			self.attention_w1 = tf.get_variable(name='attention_w1',shape=[2*self.hidden_dim,1])
			word_alpha = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(all_output_words,[-1,2*self.hidden_dim]),self.attention_w1),[self.batch_sen_num,-1]),1),[self.batch_sen_num,1,-1])
			all_output_sens = tf.reshape(tf.matmul(word_alpha,all_output_words),[-1,2*self.hidden_dim])
			all_output_sens = tf.reshape(all_output_sens,[-1,self.sen_num,2*self.hidden_dim])

		if elmo_type in ['word','char']:
			if elmo_type == 'word':
				options_file = Params.elmo_word_options_file
				weight_file = Params.elmo_word_weight_file
				embed_file = Params.elmo_word_embed_file
			else:
				options_file = Params.elmo_char_options_file
				weight_file = Params.elmo_char_weight_file
				embed_file = Params.elmo_char_embed_file

			bilm = BidirectionalLanguageModel(options_file,weight_file,use_character_inputs=False,embedding_weight_file=embed_file,max_batch_size=self.batch_sen_num)
			bilm_embedding_op = bilm(self.input_word_ELMO)
			bilm_embedding = weight_layers('output',bilm_embedding_op,l2_coef=0.0)
			bilm_representation = bilm_embedding['weighted_op']
			bilm_representation = tf.nn.dropout(bilm_representation,self.keep_prob)

			layer2_forward = self.LSTM()
			layer2_backward = self.LSTM()

			with tf.variable_scope('LSTM_ELMO'):
				elmo_output_words, _ = tf.nn.bidirectional_dynamic_rnn(layer2_forward,layer2_backward,bilm_representation,dtype=tf.float32)

			elmo_output_words = tf.concat(axis=2,values=elmo_output_words)

			self.attention_w2 = tf.get_variable(name='attention_w2',shape=[2*self.hidden_dim,1])

			elmo_word_alpha = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(elmo_output_words,[-1,2*self.hidden_dim]),self.attention_w2),[self.batch_sen_num,-1]),1),[self.batch_sen_num,1,-1])
			elmo_output_sens = tf.reshape(tf.matmul(elmo_word_alpha,elmo_output_words),[-1,2*self.hidden_dim])
			elmo_output_sens = tf.reshape(elmo_output_sens,[-1,self.sen_num,2*self.hidden_dim])

		if feature_type!=None and elmo_type!=None:
			all_output_sens = tf.concat(axis=2,values=[all_output_sens,elmo_output_sens])
		elif elmo_type!=None:
			all_output_sens = elmo_output_sens


		all_output_sens = tf.nn.dropout(all_output_sens,self.keep_prob)

		layer3_forward = self.LSTM()
		layer3_backward = self.LSTM()
		
		with tf.variable_scope('LSTM-SEN'):
			final_output_sens, _ = tf.nn.bidirectional_dynamic_rnn(layer3_forward,layer3_backward,all_output_sens,dtype=tf.float32)

		final_output_sens = tf.concat(axis=2,values=final_output_sens)

		self.attention_w3 = tf.get_variable(name='attention_w3',shape=[2*self.hidden_dim,1])

		sen_alpha = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(final_output_sens,[-1,2*self.hidden_dim]),self.attention_w3),[-1,self.sen_num]),1),[-1,1,self.sen_num])
		self.doc_rep = tf.reshape(tf.matmul(sen_alpha,final_output_sens),[-1,2*self.hidden_dim])

		self.doc_rep = tf.nn.dropout(self.doc_rep,self.keep_prob)

		out = tf.layers.dense(self.doc_rep,self.num_classes,use_bias=True,activation=None)
		

		self.prob = tf.nn.softmax(out,1)
		self.prediction=tf.argmax(self.prob,1,name="prediction")
		self.accuracy = tf.cast(tf.equal(self.prediction,tf.argmax(self.input_label,1)),"float")

		self.classfier_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=self.input_label))
		self._classfier_train_op = tf.train.AdamOptimizer(self.lr).minimize(self.classfier_loss)


	def LSTM(self,layers=1):
		lstms = []

		for num in range(layers):

			lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, forget_bias=1.0)
			# lstm = tf.contrib.rnn.GRUCell(self.hidden_dim)
			lstm = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=self.keep_prob)
			lstms.append(lstm)

		lstms = tf.contrib.rnn.MultiRNNCell(lstms)
		return lstms
