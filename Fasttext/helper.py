# -*- coding:utf-8-*-
import numpy as np
import random,os,math
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.keyedvectors import KeyedVectors
import multiprocessing
import time
import pickle 
from collections import Counter
from functools import wraps
import nltk
from nltk.corpus import stopwords
from numpy.random import seed
import math

seed(1234)

stopwords=stopwords.words("english")
def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print("%s runed %.2f seconds" % (func.__name__, delta))
        return ret
    return _deco


def cut(sentence):

    tokens =sentence.lower().split()
    # tokens = [word for word in sentence.split() if word not in stopwords] 
    return tokens



def get_alphabet(corpuses):
    """
    obtain the dict
            :param corpuses: 
    """
    word_counter = Counter()

    for corpus in corpuses:
        for sentence in corpus["question"].unique():
            tokens = cut(sentence)
            for token in tokens:
                word_counter[token] += 1
    print("there are {} words in dict".format(len(word_counter)))
    word_dict = {word: e + 2 for e, word in enumerate(list(word_counter))}
    word_dict['UNK'] = 1
    word_dict['<PAD>'] = 0

    return word_dict

def get_embedding(alphabet, filename="", embedding_size=100):
    """
    docstring here
        :param alphabet: word_dict of the train_dataset
        :param filename="": filename of the embedding
        :param embedding_size=100: embedding_size
    """
    embedding = np.random.rand(len(alphabet), embedding_size)
    with open(filename, encoding='utf-8') as f:
        i = 0
        for line in f:
            i += 1
            if i % 100000 == 0:
                print('epch %d' % i)
            items = line.strip().split(' ')
            if len(items) == 2:
                vocab_size, embedding_size = items[0], items[1]
                print((vocab_size, embedding_size))
            else:
                word = items[0]
                if word in alphabet:
                    embedding[alphabet[word]] = items[1:]

    print('done')

    return embedding

@log_time_delta
def prepare(corpuses,dim = 50):
    """
    docstring here
        :param corpuses: dataset
        :param dim=50: embedding dimension
    """
#   get the word_dict
    alphabet = get_alphabet(corpuses)
    embedding_file = '/Users/zhansu/program/code/embedding/glove.6B/glove.6B.50d.txt' 
    sub_embeddings = get_embedding(alphabet,filename = embedding_file,embedding_size = 50) # get the embedding of the dataset
    embedding_complex = getSubVectors_complex_random(alphabet,1) # get the complex embedding of the dataset
    return alphabet, sub_embeddings,embedding_complex
    
def load_text_vec_complex(alphabet,filename="",datafile='',embedding_size = 100):
    vectors = {}
    embedding_alphabet=[]
    file1=pd.read_csv(filename,sep='\t',names=["word","id"])
    file2=np.load(datafile)
    for i in range(len(file1)):
        word = file1['word'][i]
        if word in alphabet:
            #vectors={'vocab':word,'vec':file["number"][i]}
            vectors[word] = file2[i].astype(np.float)
            embedding_alphabet.append(word)
    return vectors,embedding_alphabet

def get_lookup_table(embedding_params):
    id2word = embedding_params['id2word']
    word_vec = embedding_params['word_vec']
    lookup_table = []

    # Index 0 corresponds to nothing
    lookup_table.append([0]* embedding_params['wvec_dim'])
    for i in range(1, len(id2word)):
        word = id2word[i]
        wvec = [0]* embedding_params['wvec_dim']
        if word in word_vec:
            wvec = word_vec[word]
        # print(wvec)
        lookup_table.append(wvec)

    lookup_table = np.asarray(lookup_table)
    return(lookup_table)

def getSubVectors_complex_random(vocab, dim=1):
    embedding = np.zeros((len(vocab), 1))
    for word in vocab:#RandomUniform(minval=0, maxval=2*math.pi)
        embedding[vocab[word]] = np.ones(1)
    return embedding


def getSubVectors_complex_uniform(max_sentence, dim=50):
    embedding = np.zeros((max_sentence, dim))
    for i in range(max_sentence):
        embedding[i] = np.random.uniform(+((2*math.pi)/30)*i, +((2*math.pi)/30)*(i+1), dim)
    return embedding
def load_text_vec(alphabet, filename="", embedding_size=100):
    vectors = {}
    with open(filename) as f:
        i = 0
        for line in f:
            i += 1
            if i % 100000 == 0:
                print ('epch %d' % i)
            items = line.strip().split(' ')
            if len(items) == 2:
                vocab_size, embedding_size = items[0], items[1]
                print (vocab_size, embedding_size)
            else:
                word = items[0]
                if word in alphabet:
                    vectors[word] = items[1:]
    print ('embedding_size', embedding_size)
    print ('done')
    print ('words found in wor2vec embedding ', len(vectors.keys()))
    return vectors

def position_index(sentence, length):
    index = np.zeros(length)
    raw_len = len(cut(sentence))
    index[:min(raw_len, length)] = range(1, min(raw_len + 1, length + 1))
    # print index
    return index

def convert_to_word_ids(sentence,alphabet,max_len = 40):
    """
    docstring here
        :param sentence: 
        :param alphabet: 
        :param max_len=40: 
    """
    indices = []
    tokens = cut(sentence)
    
    for word in tokens:
        if word in alphabet:
            indices.append(alphabet[word])
        else:
            continue
    result = indices + [alphabet['<PAD>']] * (max_len - len(indices))

    return result[:max_len]



def load(data_dir):
    """
    docstring here
    loading the dataset
        :param data_dir: the data dir
    """
    train_df = pd.read_csv(os.path.join(data_dir,'train.csv'),header = None,sep="\t",names=["question","flag"],quoting =3).fillna("WASHINGTON")
    dev_df = pd.read_csv(os.path.join(data_dir,'dev.csv'),header = None,sep="\t",names=["question","flag"],quoting =3).fillna("WASHINGTON")
    test_df = pd.read_csv(os.path.join(data_dir,'test.csv'),header = None,sep="\t",names=["question","flag"],quoting =3).fillna("WASHINGTON")
    return train_df,dev_df,test_df

def gen_with_pair_single(df, alphabet, sen_len):
    """
    docstring here get the single batch of dataset
        :param df: dataset
        :param alphabet: word_dict
        :param sen_len: sentence length
    """
    pairs = []
    for _, row in df.iterrows():
      sentence_indice = convert_to_word_ids(
          row['question'], alphabet, max_len = sen_len)
      postion_inf = position_index(row['question'], sen_len)
      label = transform(row["flag"])

      pairs.append((sentence_indice,label,postion_inf))
    return pairs
def batch_iter(data, batch_size, alphabet,shuffle = False,sen_len = 33):
    """
    docstring here
        :param data: dataset
        :param batch_size: batch_size
        :param alphabet: word_dict
        :param shuffle=False: 
        :param sen_len=33: 
    """
    data = gen_with_pair_single(
        data, alphabet, sen_len)
  
    data = np.array(data)
    data_size = len(data)

    if shuffle:
      shuffle_indice = np.random.permutation(np.arange(data_size))
      data = data[shuffle_indice]

    num_batch = int((data_size - 1) / float(batch_size)) + 1

    for i in range(num_batch):
      start_index = i * batch_size
      end_index = min((i + 1) * batch_size, data_size)

      yield data[start_index:end_index]
      
def transform(flag):
    if flag == 1:
        return [0, 1]
    else:
        return [1, 0]