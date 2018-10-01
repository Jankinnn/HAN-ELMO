import numpy as np
import os
import sys
sys.path.append('./')
import tensorflow.contrib.keras as kr
import glob
from config import Params
from bilm.data import TokenBatcher

def loadWordVec(mode='word'):
	wordVec = []
	wordMap = {}
	if mode == 'word':
		fr = open(Params.pre_word_embed_file,'r')
	else:
		fr = open(Params.pre_word_embed_file,'r')

	word_num, word_dim = map(int,fr.readline().strip().split())

	wordMap['BLANK'] = len(wordMap)
	wordMap['UNK'] = len(wordMap)
	wordVec.append([0.0]*word_dim)
	wordVec.append([0.0]*word_dim)

	while True:
		line  = fr.readline()
		if not line:
			break
		line = line.strip().split()
		wordMap[line[0]] = len(wordMap)
		wordVec.append(list(map(float,line[1:])))
	fr.close()
	wordVec = np.array(wordVec,dtype=np.float32)
	return wordMap,wordVec

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

def loadData(wordMap,relation2id,max_len=70,max_num=20,text_type='char'):
	if text_type == 'word':
		fr = open(Params.train_seg_file,'r')
		vocab_file =  Params.elmo_word_vocabs_file
	else:
		fr = open(Params.train_file,'r')
		vocab_file =  Params.elmo_char_vocabs_file

	
	batcher = TokenBatcher(vocab_file)


	train_data = []
	train_label = []
	train_name = []
	train_data_ELMO = []
	max_sens = 0
	all_types = {}

	total_sens = 20

	while True:
		line = fr.readline()
		if not line:
			break
		name, _, types = line.strip().split('\t')
		train_name.append(name.strip())
		label = [0]*len(relation2id)
		label[relation2id[types]] = 1
		train_label.append(label)

		data = []
		data_ELMO = []
		cur_sens=0

		while True:
			line = fr.readline()
			if not line:
				break
			if line.strip()=='':
				break
			if len(line.strip().split('\t')) !=2:
				continue
			line = line.strip().split('\t')[1]
			if cur_sens>=max_num:
				continue
			cur_sens +=1
			
			sent = []
			if text_type == 'word':
				for c in line.split()[:max_len]:
					if c in wordMap:
						sent.append(wordMap[c])
					else:
						sent.append(wordMap['UNK'])
				while len(sent) < max_len:
					sent.append(wordMap['BLANK'])

				elmo_sent = list(line.strip().split())[:max_len]
			else:
				for c in list(line)[:max_len]:
					if c in wordMap:
						sent.append(wordMap[c])
					else:
						sent.append(wordMap['UNK'])
				while len(sent) < max_len:
					sent.append(wordMap['BLANK'])
				elmo_sent = list(line.strip())[:max_len]


			data.append(sent)
			data_ELMO.append(elmo_sent)

		while cur_sens<max_num:
			cur_sens+=1
			data.append([0]*max_len)
			data_ELMO.append(['<UNK>'])

		train_data += data 
		train_data_ELMO += data_ELMO
		
	train_data = np.array(train_data,dtype=np.int32)
	train_data_ELMO = batcher.batch_sentences(train_data_ELMO)
	
	train_data = np.reshape(train_data,(-1,max_num,max_len))
	train_data_ELMO = np.reshape(train_data_ELMO,(-1,max_num,max_len+2))
	train_label = np.array(train_label,dtype=np.float32)

	return train_data,train_data_ELMO,train_name,train_label

def loadTest(wordMap,relation2id,max_len=70,max_num=20,text_type='char'):
	if text_type == 'word':
		fr = open(Params.test_seg_file,'r')
		vocab_file =  Params.elmo_word_vocabs_file
	else:
		fr = open(Params.test_file,'r')
		vocab_file =  Params.elmo_char_vocabs_file

	
	batcher = TokenBatcher(vocab_file)


	train_data = []

	train_name = []
	train_data_ELMO = []



	while True:
		line = fr.readline()
		if not line:
			break
		name = line.strip()
		train_name.append(name.strip())

		data = []
		data_ELMO = []
		cur_sens=0

		while True:
			line = fr.readline()
			if not line:
				break
			if line.strip()=='':
				break
			if len(line.strip().split('\t')) !=2:
				continue
			line = line.strip().split('\t')[1]
			if cur_sens>=max_num:
				continue
			cur_sens +=1
			
			sent = []
			if text_type == 'word':
				for c in line.split()[:max_len]:
					if c in wordMap:
						sent.append(wordMap[c])
					else:
						sent.append(wordMap['UNK'])
				while len(sent) < max_len:
					sent.append(wordMap['BLANK'])

				elmo_sent = list(line.strip().split())[:max_len]
			else:
				for c in list(line)[:max_len]:
					if c in wordMap:
						sent.append(wordMap[c])
					else:
						sent.append(wordMap['UNK'])
				while len(sent) < max_len:
					sent.append(wordMap['BLANK'])
				elmo_sent = list(line.strip())[:max_len]


			data.append(sent)
			data_ELMO.append(elmo_sent)

		while cur_sens<max_num:
			cur_sens+=1
			data.append([0]*max_len)
			data_ELMO.append(['<UNK>'])

		train_data += data 
		train_data_ELMO += data_ELMO
		
	train_data = np.array(train_data,dtype=np.int32)
	train_data_ELMO = batcher.batch_sentences(train_data_ELMO)
	train_data = np.reshape(train_data,(-1,max_num,max_len))
	train_data_ELMO = np.reshape(train_data_ELMO,(-1,max_num,max_len+2))

	return train_data,train_data_ELMO,train_name