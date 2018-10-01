import os
class Params:
	num_epochs = 10
	lr = 0.001
	Kfold = 10
	
	data_dir = './data'
	model_dir = './model'
	result_dir='./result'


	elmo_word_options_file = os.path.join(data_dir , 'word_options.json')
	elmo_word_weight_file = os.path.join(data_dir ,'word_weights.hdf5')
	elmo_word_embed_file = os.path.join(data_dir , 'word_embedding.hdf5')
	elmo_word_vocabs_file = os.path.join(data_dir,'vocabs.word.txt')

	elmo_char_options_file = os.path.join(data_dir , 'char_options.json')
	elmo_char_weight_file = os.path.join(data_dir ,'char_weights.hdf5')
	elmo_char_embed_file = os.path.join(data_dir , 'char_embedding.hdf5')
	elmo_char_vocabs_file = os.path.join(data_dir,'vocabs.char.txt')



	pre_word_embed_file = os.path.join(data_dir, 'callreason-word-300d')
	pre_char_embed_file = os.path.join(data_dir, 'callreason-char-300d')

	relation_file = os.path.join(data_dir, 'relations.txt')
	relation_hyper_file = os.path.join(data_dir, 'relations_hyper.txt')

	train_file = os.path.join(data_dir, 'callreason.train.0903')
	train_seg_file = os.path.join(data_dir,'callreason.train.seg.0903')

	test_file = os.path.join(data_dir, 'callreason.testD.0925')
	test_seg_file = os.path.join(data_dir,'callreason.testD.seg.0925')

	stacking_result_file = os.path.join(result_dir,'stacking.result.txt')

	sen_max_len = 70
	doc_max_sen = 20

	word_dim = 300

	hidden_dim = 300
	batch_size = 50
	num_classes = 42
	dropout = 0.5








