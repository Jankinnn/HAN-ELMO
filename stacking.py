import sys
import os
import numpy as np
import pickle as cp
import tensorflow as tf
from glob import glob
import datetime
from config import Params
from collections import Counter

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('gpu','0','gpu id')



def loadRelations():
	relation2id = {}
	fr = open(Params.relation_file,'r')
	while True:
		line = fr.readline()
		if not line:
			break
		rel = line.strip()
		relation2id[rel] = len(relation2id)
	fr.close()
	return relation2id

def getDirs(dirs,root=Params.result_dir):

	dir_ = [files for files in glob(os.path.join(root,'*')) if os.path.isdir(files)]
	for d in dir_:
		dirs = getDirs(dirs,d)
	dirs += dir_
	return dirs



def getLabel():
	relation2id = loadRelations()

	fr = open(Params.train_file,'r')

	train_label = []
	while True:
		line = fr.readline()
		if not line:
			break
		line = line.strip()

		if len(line)<=0:
			continue
		if line[0] in '|/':
			
			name= line.strip().split('\t')
			label = [0]*len(relation2id)
			if name[2] not in relation2id:
				print('label error')
			label[relation2id[name[2]]] = 1
			train_label.append(label)

			data = []
			while True:
				line = fr.readline()
				if not line:
					break
				if line.strip()=='':
					break			
	fr.close()
	train_label = np.array(train_label,dtype=np.float32)
	return train_label

def getFiles(dirs, mode='_oof_'):
	all_files = []
	for d in dirs:
		all_files += glob(os.path.join(d,'*'+mode+'*'))
	return all_files

def write_result(all_pred, id2relation):
	son_father = {}
	f = open(Params.relation_hyper_file,'r')
	lines = f.readlines()
	for line in lines:
		if line.strip()=='':
			continue
		line = line.strip().split()
		son_father[line[1]] = line[0]
	f.close()


	fr = open(Params.test_file,'r')
	fw = open(Params.stacking_result_file,'w')

	idx=0
	while True:
		line = fr.readline()
		if not line:
			break
		
		line = line.strip()
		if len(line)<=0:
			continue
		if line[0] in '|/':
			name = line.strip()
			types = id2relation[Counter(all_pred[idx].tolist()).most_common(1)[0][0]]
			father = son_father[types]
			fw.write(name+'\t'+father+'\t'+types+'\n')
			
			while True:
				line = fr.readline()
				if not line:
					break
				if line.strip()=='':
					break		
	fw.close()
	fr.close()
	fr2.close()

train_data = np.empty((20000,0))

dirs = [Params.result_dir]
all_files = getFiles(getDirs(dirs))
for file in all_files:
	with open(file,'rb') as fr:
		jzdata = cp.load(fr)
		train_data = np.concatenate((train_data,jzdata),axis=1)

# test data
test_data = np.empty((10000,0))

for file in all_files:
	file = file.replace('_oof_','_pre_')
	with open(file,'rb') as fr:
		jzdata = cp.load(fr)[:10000]
		test_data = np.concatenate((test_data,jzdata),axis=1)

train_label = getLabel()

dim = train_data.shape[1]

print(dim)
print(train_label.shape[1])



kfold=10
kfold_size = train_data.shape[0] // kfold
order = list(range(train_data.shape[0]))
np.random.shuffle(order)
batch_size = 200


id2relation = {}
relation2id = loadRelations()
for k,v in relation2id.items():
	id2relation[v] = k

avg_acc = 0.0
all_pre = []

for k in list(range(kfold)):

	tf.reset_default_graph()

	input_x = tf.placeholder(dtype=tf.float32,shape=[None,dim], name='input_x')
	input_y = tf.placeholder(dtype=tf.float32,shape=[None,42],name='input_y')
	is_training = tf.placeholder(dtype=tf.bool,name='is_training')
	keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')
	input_x = tf.nn.dropout(input_x,keep_prob)




	layer1 = tf.layers.dense(input_x,300,use_bias=True,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())

	layer1_d = tf.nn.dropout(layer1,keep_prob)



	layer2 = tf.layers.dense(layer1_d,300,use_bias=True,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
	w1 = tf.layers.dense(tf.concat(axis=1,values=[layer2,layer1]),300,use_bias=True,activation=tf.nn.sigmoid,kernel_initializer=tf.contrib.layers.xavier_initializer())


	layer2 = w1 * layer2 + (1-w1)*layer1

	layer2_d = tf.nn.dropout(layer2,keep_prob)



	layer3 = tf.layers.dense(layer2_d,300,use_bias=True,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())

	w2 = tf.layers.dense(tf.concat(axis=1,values=[layer3,layer2]),300,use_bias=True,activation=tf.nn.sigmoid,kernel_initializer=tf.contrib.layers.xavier_initializer())

	layer3 = w2 * layer3 + (1-w2)*layer2

	layer3_d = tf.nn.dropout(layer3,keep_prob)





	out = tf.layers.dense(layer3_d,42,use_bias=True,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer())
	classfier_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=input_y))

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		_classfier_train_op = tf.train.AdamOptimizer(0.001).minimize(classfier_loss)

	prob = tf.nn.softmax(out,1)
	prediction=tf.argmax(prob,1,name="prediction")
	accuracy = tf.cast(tf.equal(prediction,tf.argmax(input_y,1)),"float")
	gpu_options = tf.GPUOptions(visible_device_list=FLAGS.gpu,allow_growth=True)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))
	saver = tf.train.Saver(max_to_keep=None)
	sess.run(tf.global_variables_initializer())

	dev_index = order[k*kfold_size:(k+1)*kfold_size]
	train_index = order[:k*kfold_size] + order[(k+1)*kfold_size:]

	train_set = train_data[train_index]
	dev_set = train_data[dev_index]
	train_y = train_label[train_index]
	dev_y = train_label[dev_index]

	total_step = 0
	max_acc = 0.0
	for epoch in range(40):
		tmp_order = list(range(train_set.shape[0]))
		np.random.shuffle(tmp_order)
		for i in range(train_set.shape[0]//batch_size):
			feed_dict = {}
			feed_dict[is_training]=True
			feed_dict[keep_prob]=0.5

			feed_dict[input_x] = train_set[tmp_order[i*batch_size:(i+1)*batch_size]]
			feed_dict[input_y] = train_y[tmp_order[i*batch_size:(i+1)*batch_size]]
			_,acc,loss = sess.run([_classfier_train_op,accuracy,classfier_loss],feed_dict)
			acc = np.mean(acc)
			total_step+=1
			
			time_str = datetime.datetime.now().isoformat()

			if total_step%50==0:
				feed_dict = {}
				feed_dict[input_x] = dev_set 
				feed_dict[is_training]=False
				feed_dict[keep_prob]=1.0


				feed_dict[input_y] = dev_y
				acc = sess.run([accuracy],feed_dict)
				acc = np.mean(acc)
				if acc>max_acc:
					max_acc = acc
					print('accuracy: %f' %acc)
					print('saving model')
					path = saver.save(sess,'./model/callreason_stacking.'+str(k),global_step=0)
					tempstr = 'have saved model to ' + path
					print(tempstr)

	pathname='./model/callreason_stacking.'+str(k)+'-0'
	print('load model:'+pathname)
	try:
		saver = tf.train.Saver()
		saver.restore(sess,pathname)
	except:
		exit()

	print('end load model')

	print('best dev test:')
	feed_dict = {}
	feed_dict[input_x] = dev_set 
	feed_dict[is_training]=False
	feed_dict[keep_prob]=1.0


	feed_dict[input_y] = dev_y
	acc = sess.run([accuracy],feed_dict)
	acc = np.mean(acc)
	
	print('accuracy: %f' %acc)

	feed_dict = {}
	feed_dict = {}
	feed_dict[input_x] = test_data 
	feed_dict[is_training]=False
	feed_dict[keep_prob]=1.0


	pre = sess.run([prediction],feed_dict)
	pre = np.reshape(pre,(-1,1))
	all_pre.append(pre)

	avg_acc+=max_acc
print(avg_acc/kfold)
all_pre = np.concatenate(all_pre,axis=1)
write_result(all_pre,id2relation)












