import tensorflow as tf
import numpy as np
import time
import datetime
import os
from init import *
from model import *
from config import Params
import sys
sys.path.append('./')
import random
from bilm.data import TokenBatcher
import pickle as pk


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('gpu','0','gpu id')
tf.app.flags.DEFINE_string('text_type','char','char or word')
tf.app.flags.DEFINE_string('prt','1','use pretrained embedding')
tf.app.flags.DEFINE_string('elmo','1','use elmo embedding')





def main(_):
	source = Params.data_dir
	target = Params.model_dir

	dropout = Params.dropout
	batch_size = Params.batch_size

	
	train_times = Params.num_epochs
	sen_len = Params.sen_max_len
	sen_num = Params.doc_max_sen

	text_type = FLAGS.text_type
	if text_type in ['word','char']:
		wordMap, wordVec = loadWordVec(text_type)
	else:
		exit(0)

	if FLAGS.prt == '1':
		feature_type = text_type
	else:
		feature_type = None

	if FLAGS.elmo == '1':
		elmo_type = text_type
	else:
		elmo_type = None

	if feature_type is None and elmo_type is None:
		exit(0)


	relation2id = loadRelations()

	train_data, train_data_ELMO,train_name, train_label = loadData(wordMap,relation2id,max_len=sen_len,max_num = sen_num,text_type=text_type)
	test_data, test_data_ELMO,test_name = loadTest(wordMap,relation2id,max_len=sen_len,max_num = sen_num,text_type=text_type)

	

	total_dev_probs = np.zeros((train_data.shape[0],len(relation2id)),dtype=np.float32)
	total_test_probs = 0.0

	final_accuracy = 0.0
	


	gpu_options = tf.GPUOptions(visible_device_list=FLAGS.gpu,allow_growth=True)
	with tf.Graph().as_default():
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))
		with sess.as_default():

			initializer = tf.contrib.layers.xavier_initializer()
			with tf.variable_scope('',initializer=initializer):
				m = HAN(feature_type=feature_type,elmo_type=elmo_type,pre_embed=wordVec)

			
				
			def train_step(word_batch,word_batch_ELMO,label_batch,global_step):
				feed_dict={}
				
				feed_dict[m.keep_prob] = dropout
				feed_dict[m.input_word] = np.reshape(word_batch,(-1,sen_len))
				feed_dict[m.input_word_ELMO] = np.reshape(word_batch_ELMO,(-1,sen_len+2))
			
				feed_dict[m.input_label] = label_batch

				_,classfier_loss,accuracy = sess.run([m._classfier_train_op,m.classfier_loss,m.accuracy],feed_dict)

				accuracy=np.reshape(np.array(accuracy),(-1))
				acc=np.mean(accuracy)
				time_str = datetime.datetime.now().isoformat()

				global_step+=1
				if global_step % 10 == 0:
					tempstr = "{}: step {}, classifier_loss {:g}, acc {:g}".format(time_str, global_step, classfier_loss,acc)
					print(tempstr)
				return global_step

			def dev_step(word_batch,word_batch_ELMO,label_batch):
				

				feed_dict={}

				feed_dict[m.keep_prob] = 1.0

				feed_dict[m.input_word] = np.reshape(word_batch,(-1,sen_len))
				feed_dict[m.input_label] = label_batch
				feed_dict[m.input_word_ELMO] = np.reshape(word_batch_ELMO,(-1,sen_len+2))


				accuracy = sess.run([m.accuracy],feed_dict)

				return accuracy

			def test_step(word_batch,word_batch_ELMO):
				feed_dict={}
				
				feed_dict[m.keep_prob] = 1.0

				feed_dict[m.input_word] = np.reshape(word_batch,(-1,sen_len))
				feed_dict[m.input_word_ELMO] = np.reshape(word_batch_ELMO,(-1,sen_len+2))


				predict,prob = sess.run([m.prediction,m.prob],feed_dict)

				return predict,prob

			def dev(devset,devset_ELMO,devset_label):
				pos = 0

				total_accuracy = []
				lens = devset.shape[0]
				while pos<lens:
					if pos+batch_size>lens:
						break
					accuracy = dev_step(devset[pos:pos+batch_size],devset_ELMO[pos:pos+batch_size],devset_label[pos:pos+batch_size])
					pos += batch_size
					total_accuracy.append(accuracy)

				if pos<lens:
					accuracy= dev_step(devset[-batch_size:],devset_ELMO[-batch_size:],devset_label[-batch_size:])
					total_accuracy.append(accuracy[-lens+pos:])

				total_accuracy = np.mean(np.concatenate(total_accuracy,axis=0))
				return total_accuracy

			def test(test_data,test_data_ELMO):
				pos=0


				all_prob = []
				lens = test_data.shape[0]
				while pos<lens:
					if pos+batch_size>lens:
						break

					predic,prob = test_step(test_data[pos:pos+batch_size],test_data_ELMO[pos:pos+batch_size])
					all_prob.append(np.reshape(prob,(-1,len(relation2id))))
					pos += batch_size


				if pos<lens:
					predic,prob= test_step(test_data[-batch_size:],test_data_ELMO[-batch_size:])
					all_prob.append(np.reshape(prob[-lens+pos:],(-1,len(relation2id))))
					

				all_prob = np.concatenate(all_prob,axis=0)

				return all_prob


		
			order = list(range(train_data.shape[0]))
			np.random.shuffle(order)

			fold_size = len(train_data) // Params.Kfold + 1

			for k in range(Params.Kfold):
				print('CV: %d/%d' % (k+1, Params.Kfold))

				global_step = 0

				max_accuracy = 0.0
				
				sess.run(tf.global_variables_initializer())
				saver = tf.train.Saver(max_to_keep=None)


				dev_index = order[k*fold_size:(k+1)*fold_size]
				train_index = order[:k*fold_size] + order[(k+1)*fold_size:]

				trainset, trainset_ELMO, trainset_label = train_data[train_index], train_data_ELMO[train_index], train_label[train_index]
				devset, devset_ELMO, devset_label = train_data[dev_index], train_data_ELMO[dev_index], train_label[dev_index]

				

				for epoch in range(train_times):
					print('epoch: %d' % epoch)

					trainset_order = list(range(trainset.shape[0]))
					np.random.shuffle(trainset_order)

					batch_num = trainset.shape[0] // batch_size
					for i in range(batch_num):
						idx = trainset_order[i*batch_size:(i+1)*batch_size]
						global_step = train_step(trainset[idx],trainset_ELMO[idx],trainset_label[idx],global_step)

						if global_step % 50 == 0:
							acc = dev(devset,devset_ELMO,devset_label)
							if acc > max_accuracy:
								max_accuracy = acc
								print('accuracy: %f' %acc)
								print('saving model')
								path = saver.save(sess,target+'/callreason_hna_model.%s.prt%s.elmo%s.cv.%d'%(FLAGS.text_type,FLAGS.prt,FLAGS.elmo,k),global_step=0)
								tempstr = 'have saved model to ' + path
								print(tempstr)
				final_accuracy += max_accuracy

				pathname=target+'/callreason_hna_model.%s.prt%s.elmo%s.cv.%d'%(FLAGS.text_type,FLAGS.prt,FLAGS.elmo,k)+'-0'
				print('load model:'+pathname)
				try:
					saver.restore(sess,pathname)
				except:
					exit()
				print('end load model')

				all_dev_prob = test(devset,devset_ELMO)


				for i in range(devset.shape[0]):
					total_dev_probs[dev_index[i]:] = all_dev_prob[i]


				all_test_prob = test(test_data,test_data_ELMO)
				total_test_probs = total_test_probs + all_test_prob/float(Params.Kfold)

			print('final accuracy: %f' % (final_accuracy/float(Params.Kfold)))

			with open(Params.result_dir+'/callreason_oof_.%s.prt%s.elmo%s.cv.%d.pkl'%(FLAGS.text_type,FLAGS.prt,FLAGS.elmo,Params.Kfold),'wb') as fw:
				pk.dump(total_dev_probs,fw)

			with open(Params.result_dir+'/callreason_pre_.%s.prt%s.elmo%s.cv.%d.pkl'%(FLAGS.text_type,FLAGS.prt,FLAGS.elmo,Params.Kfold),'wb') as fw:
				pk.dump(total_test_probs,fw)



if __name__ == '__main__':
	tf.app.run()