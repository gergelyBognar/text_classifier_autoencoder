import numpy as np
import random
import pickle
import time
import csv
import tensorflow as tf

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')


##############################
#data preprocessing functions#
##############################

#remove some further undesirable characters
def remove_some_chars(text):
    text = text.replace('\xe2\x80\x98', '').replace('\xe2\x80\x99', '').replace('\xe2\x80\x9a', '').replace('\xe2\x80\x9b', '').replace('\xe2\x80\x9c', '').replace('\xe2\x80\x9d', '').replace('\xe2\x80\x9e', '').replace('\xe2\x80\x9f', '') #remove unicode quotion mark characters
    chars_to_remove = set(['[', ']', '(', ')', '<', '>', '{', '}', "'", '"', "`", '!', '?', ':', ';', ',', '|', '@'])
    #keepers: '-.&#*+='
    return ''.join([i for i in text if i not in chars_to_remove])

#stripdown sentence
def stripdown_sentence(sentence):
    sentence = sentence.lower()
    sentence = remove_some_chars(sentence)
    return sentence


##################################
#embedding and dictionary loading#
##################################

#load embeddings dictionary and reverse dictionary
def dictionary_embedding():
    with open('./data/dictionary_embedding.pickle') as f:
        final_embeddings, dictionary, reverse_dictionary = pickle.load(f)
        return final_embeddings, dictionary, reverse_dictionary


#############################
#classification data reading#
#############################

#with que reader
def read_and_decode_csv_class(file_training):
    #training data
    filename_queue = tf.train.string_input_producer([file_training])

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [['alma'], [2.0], [2.0], [2.0], [2.0], [2.0]]
    col1, col2, col3, col4, col5, col6 = tf.decode_csv(
        value, record_defaults=record_defaults)
    features = col1
    label = tf.stack([col2, col3, col4, col5, col6])
    
    return features, label

def training_data_class():
    file_name = './data/music_news_dataset_labelled_training.csv'
    x, y_ = read_and_decode_csv_class(file_name)
    return x, y_

def test_data_class():
    file_name = './data/music_news_dataset_labelled_test.csv'
    x, y_ = read_and_decode_csv_class(file_name)
    return x, y_


##########################
#autoencoder data reading#
##########################

#with que reader
def training_data_autoencoder():
    file_name = './data/gigers_news_dataset.csv'

    #training data
    filename_queue = tf.train.string_input_producer([file_name])

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [[0], ['title'], ['date']]
    col1, col2, col3 = tf.decode_csv(
        value, record_defaults=record_defaults)
    ids = col1
    titles = col2
    
    return titles


################
#for evaluation#
################

#without que reader
def gigers_data_list():
    file_name = './data/gigers_news_dataset.csv'

    ids = list()
    titles = list()
    with open(file_name, 'rb') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            ids.append(row[0])
            titles.append(row[1])
    
    return ids, titles

def write_gigers_data_classified(ids, titles, classes):
    file_name = './data/gigers_news_dataset_classified.csv'
    
    with open(file_name, 'wb') as csv_file:
        csv_writer = csv.writer(csv_file)

        for index, id_gigers in enumerate(ids):
            csv_writer.writerow([id_gigers, titles[index], classes[index]])

def write_gigers_data_encoded(ids, titles, codes):
    with open('./data/gigers_dataset_encoded.pickle', 'wb') as f:
        pickle.dump((ids, titles, codes), f)

def read_gigers_data_encoded():
    with open('./data/gigers_dataset_encoded.pickle') as f:
        ids, titles, codes = pickle.load(f)
        return ids, titles, codes

