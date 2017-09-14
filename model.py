import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import loader
import numpy as np
import time

from data import data_loader

def trainer_run(
    max_sentence_length_long = 37,
    max_sentence_length_short = 10,
    batch_size_training = 64,
    batch_size_serving = 100,
    batch_thread_number = 1,
    batch_capacity = 5000,
    min_after_dequeue = 1000,
    step_number = 8000,
    log_step_number_class = 50,
    accuracy_step_number_class = 500,
    log_step_number_autoencoder = 50,
    keep_prob_class = 0.5,
    keep_prob_autoencoder = 0.5,
    cell_size = 512,
    autoencoder_hidden_layer_sizes = [],
    accuracy_sample_number_training = 500,
    accuracy_sample_number_test = 500,
    is_train_model = True,
    is_transfer_learning_class = True,
    is_save_model = False,
    is_save_model_checkpoint = False,
    labels_number = 5,
    is_multilabel = False,
    learning_rate_class = 0.004,
    learning_rate_autoencoder = 0.004,
    is_autoencoder = False,
    is_tensorboard_log = True,
    optimizer = 'AdamOptimizer'):


    ###############################
    #calculated control parameters#
    ###############################

    #cell size
    if not autoencoder_hidden_layer_sizes:
        #if autoencoder hidden layer list is empty, then code is the cell of the encoder LSTM
        code_size = cell_size
    else:
        #if autoencoder hidden layer list is not empty, code is the last hidden layer
        code_size = autoencoder_hidden_layer_sizes[-1]

    #append cell size to the begining of autoencoder layer sizes list
    #because the output of the encoder cell will be the input for the autoencoder layers
    autoencoder_layer_sizes = [cell_size] + autoencoder_hidden_layer_sizes


    ###############################
    #data readers and placeholders#
    ###############################

    #load embeddings and dictionary
    embeddings, dictionary, reverse_dictionary = data_loader.dictionary_embedding()
    embeddings[0] = embeddings[0] * 0 #zero out the embedding of UNK
    vocabulary_size, embedding_size = embeddings.shape

    #map for word -> index
    word_to_id_words = tf.constant([reverse_dictionary[index] for index in range(len(reverse_dictionary))])
    word_to_id_table = tf.contrib.lookup.string_to_index_table_from_tensor(
        mapping=word_to_id_words, num_oov_buckets=0, default_value=0)


    #training input of classifier
    x_training_class, y_training_class_ = data_loader.training_data_class() #this data is already in all lowercase letters and cleared from unwanted characters
    #Bit0: other cathegory
    #Bit1: new music
    #Bit2: video
    #Bit3: announcement
    #Bit4: tour

    #test input of classifier
    x_test_class, y_test_class_ = data_loader.test_data_class()

    #training input batches from classifier data
    x_train_batch_class, y_train_batch_class_ = tf.train.shuffle_batch(
          [x_training_class, y_training_class_],
          batch_size=batch_size_training,
          num_threads=batch_thread_number,
          capacity=batch_capacity,
          min_after_dequeue=min_after_dequeue)


    #training data for the autoencoder is gigers titles
    x_train_autoencoder = data_loader.training_data_autoencoder()

    x_train_batch_autoencoder = tf.train.shuffle_batch(
          [x_train_autoencoder],
          batch_size=batch_size_training,
          num_threads=batch_thread_number,
          capacity=batch_capacity,
          min_after_dequeue=min_after_dequeue)

    
    #serving input
    x_serving = tf.placeholder(tf.string, shape=[None], name='x_serving')


    #######
    #model#
    #######


    #helper functions#
    ##################

    def split_sentence(x, is_serving=False):
        #stripdown sentence if not save serving model (tensorflow bug: serving model can't use py_func)
        if not (is_serving and is_save_model): 
            x = tf.py_func(data_loader.stripdown_sentence, [x], tf.string)

        #split sentence to words
        if is_serving:
            x_split = tf.string_split(x).values
        else:
            x_split = tf.string_split([x]).values

        return x_split    

    def sentence_to_ids(x, max_sentence_length, is_serving=False):
        #split sentence to words
        x_split = split_sentence(x, is_serving=is_serving)

        #lookup ids, based on dictionary
        x_ids = word_to_id_table.lookup(x_split, name='ids')

        #pad with zeros
        ids_padding = tf.zeros(shape=[max_sentence_length], dtype=tf.int64)
        x_ids_padded = tf.concat([x_ids, ids_padding], axis=0)

        #limit to length: max_sentence_length
        x_ids_limited = x_ids_padded[:max_sentence_length]

        return x_ids_limited

    def sentence_to_ids_short(x, is_serving=False):
        x_ids_limited = sentence_to_ids(x, max_sentence_length=max_sentence_length_short, is_serving=is_serving)
        return x_ids_limited

    def sentence_to_ids_long(x, is_serving=False):
        x_ids_limited = sentence_to_ids(x, max_sentence_length=max_sentence_length_long, is_serving=is_serving)
        return x_ids_limited

    def sentence_to_embeddings(x, is_batch=False, is_serving=False, is_short=False):
        if is_batch:
            if is_short:
                x_ids = tf.map_fn(sentence_to_ids_short, x, dtype=tf.int64)
            else:
                x_ids = tf.map_fn(sentence_to_ids_long, x, dtype=tf.int64)
        else:
            if is_short:
                x_ids = sentence_to_ids_short(x, is_serving=is_serving)
            else:
                x_ids = sentence_to_ids_long(x, is_serving=is_serving)

        x_word_ids = tf.transpose(x_ids) #shape: [[sentence1], [sentence2], ...] -> shape: [[word1_sentence_1, word1_sentence_2 ...], [word2_sentence_1, word2_sentence_2 ...]]

        x_word_embeddings = tf.nn.embedding_lookup(embeddings, x_word_ids)

        return x_word_embeddings

    def get_batch_size(is_batch=False, is_serving=False):
        if is_batch:
            #batch_size = tf.shape(x)[0]
            if is_serving:
                #batch_size = batch_size_serving[0]
                batch_size = batch_size_serving
            else:
                batch_size = batch_size_training
        else:
            batch_size = 1

        return batch_size

    #define function to restore string from the embedded sentence
    def sentence_embedded_to_string(embeddings, sentence_embedded):
        dist = np.dot(embeddings, np.transpose(sentence_embedded))
        word_indeces = np.argmax(dist, axis=0)
        sentence_string = ''
        for word_index in word_indeces:
            sentence_string += reverse_dictionary[word_index] + ' '
        return sentence_string


    #definition of layers#
    ######################

    #LSTM cell class (variables and computation)
    class LstmCell:
        def __init__(self):
            #create all trainable variables
            #input gate: input, previous output, and bias
            self.ix = tf.Variable(tf.truncated_normal([embedding_size, cell_size], -0.1, 0.1))
            self.im = tf.Variable(tf.truncated_normal([embedding_size, cell_size], -0.1, 0.1))
            self.ib = tf.Variable(tf.zeros([1, cell_size]))
            #forget gate: input, previous output, and bias
            self.fx = tf.Variable(tf.truncated_normal([embedding_size, cell_size], -0.1, 0.1))
            self.fm = tf.Variable(tf.truncated_normal([embedding_size, cell_size], -0.1, 0.1))
            self.fb = tf.Variable(tf.zeros([1, cell_size]))
            #memory cell: input, state and bias
            self.cx = tf.Variable(tf.truncated_normal([embedding_size, cell_size], -0.1, 0.1))
            self.cm = tf.Variable(tf.truncated_normal([embedding_size, cell_size], -0.1, 0.1))
            self.cb = tf.Variable(tf.zeros([1, cell_size]))
            #output gate: input, previous output, and bias
            self.ox = tf.Variable(tf.truncated_normal([embedding_size, embedding_size], -0.1, 0.1))
            self.om = tf.Variable(tf.truncated_normal([embedding_size, embedding_size], -0.1, 0.1))
            self.ob = tf.Variable(tf.zeros([1, embedding_size]))
            #variables to map states to output
            self.w = tf.Variable(tf.truncated_normal([cell_size, embedding_size], -0.1, 0.1))
            self.b = tf.Variable(tf.zeros([embedding_size]))

            #add variables to tensorboard summary
            tf.summary.histogram('ix', self.ix)
            tf.summary.histogram('im', self.im)
            tf.summary.histogram('ib', self.ib)
            tf.summary.histogram('fx', self.fx)
            tf.summary.histogram('fm', self.fm)
            tf.summary.histogram('fb', self.fb)
            tf.summary.histogram('cx', self.cx)
            tf.summary.histogram('cm', self.cm)
            tf.summary.histogram('cb', self.cb)
            tf.summary.histogram('ox', self.ox)
            tf.summary.histogram('om', self.om)
            tf.summary.histogram('ob', self.ob)
            tf.summary.histogram('w', self.w)
            tf.summary.histogram('b', self.b)

        def init_state_variables(self, batch_size = 1):
            #output and state variables (non-trainable)
            self.output = tf.Variable(tf.zeros([batch_size, embedding_size]), trainable=False, validate_shape=False)
            self.state = tf.Variable(tf.zeros([batch_size, cell_size]), trainable=False, validate_shape=False)

        #make one step
        def run(self, input_value, is_batch=False):
            if not is_batch:
                input_value = tf.reshape(input_value, [1, embedding_size])

            input_gate = tf.sigmoid(tf.matmul(input_value, self.ix) + tf.matmul(self.output, self.im) + self.ib)
            forget_gate = tf.sigmoid(tf.matmul(input_value, self.fx) + tf.matmul(self.output, self.fm) + self.fb)
            update = tf.matmul(input_value, self.cx) + tf.matmul(self.output, self.cm) + self.cb
            self.state = forget_gate * self.state + input_gate * tf.tanh(update)
            output_gate = tf.sigmoid(tf.matmul(input_value, self.ox) + tf.matmul(self.output, self.om) + self.ob)
            self.output = tf.nn.l2_normalize(output_gate * tf.nn.xw_plus_b(self.state, self.w, self.b), dim=1)
        
        def get_state(self):
            #normalize state              
            normalized_state = tf.nn.l2_normalize(self.state, dim=1)
            return normalized_state

        #for the decoder to get the state from the encoder
        def set_state(self, input_state):
            self.state = input_state

    #fully connected autoencoder layers (variables and computation)
    class Autoencoder:
        def __init__(self, autoencoder_layer_sizes):
            #tied weights between encoder and decoder
            self.w_list = list()
            self.b_list = list()
            self.b_prime_list = list()
            #weights like: [input_size -> hidden_1_size], [hidden_1_size -> hidden_2_size], ... , [hidden_n-1_size -> hidden_n_size]
            for index in range(len(autoencoder_layer_sizes) - 1):
                #generate names
                w_name = 'w_autoencoder_' + str(index)
                b_name = 'b_autoencoder_' + str(index)
                b_prime_name = 'b_prime_autoencoder_' + str(index)

                #calculate layer matrix size
                layer_input_size = autoencoder_layer_sizes[index]
                layer_output_size = autoencoder_layer_sizes[index + 1]

                #create weights and biases
                w = tf.Variable(tf.random_normal([layer_input_size, layer_output_size]), name = w_name)
                b = tf.Variable(tf.zeros([layer_output_size]), name = b_name)
                b_prime = tf.Variable(tf.zeros([layer_input_size]), name = b_prime_name)

                #add variables to tensorboard summary
                tf.summary.histogram(w_name, w)
                tf.summary.histogram(b_name, b)
                tf.summary.histogram(b_prime_name, b_prime)

                #append weights and biases to list
                self.w_list.append(w)
                self.b_list.append(b)
                self.b_prime_list.append(b_prime)

        def run_encode(self, input_value):
            #if autoencoder is empty, it lets through the tensor untouched
            code = input_value

            #apply all layers
            for index in range(len(self.w_list)):
                code = tf.nn.xw_plus_b(code, self.w_list[index], self.b_list[index])

            #code_normaized = tf.nn.l2_normalize(code, dim=1)

            return code

        def run_decode(self, code):
            #if autoencoder is empty, it lets through the tensor untouched
            output_value = code

            #apply all layers in reverse order, weights are transposed in this order
            for index in reversed(range(len(self.w_list))):
                output_value = tf.nn.xw_plus_b(output_value, tf.transpose(self.w_list[index]), self.b_prime_list[index])

            return output_value

        def run(self, input_value):
            #encode input
            code = self.run_encode(input_value)
            #dropout on code, for decoding
            code_dropout = tf.nn.dropout(code, keep_prob = keep_prob_autoencoder)
            #decode the code with dropout applied on it
            output_value = self.run_decode(code_dropout)

            #returns the output of decoder and the non-dropout code
            return output_value, code

    #variables for classification
    w_class = tf.Variable(tf.random_normal([code_size, labels_number]), name = 'w_class')
    b_class = tf.Variable(tf.random_normal([labels_number]), name = 'b_class')


    #assamble the layers#
    #####################

    #create encoder and decoder cells
    with tf.name_scope('encoder_cell'):
        encoder_cell = LstmCell()
    with tf.name_scope('decoder_cell'):
        decoder_cell = LstmCell()

    #create the fully connected autoencoder
    with tf.name_scope('autoencoder'):
        autoencoder = Autoencoder(autoencoder_layer_sizes)

    #enroll encoder, get code and embedded sentences
    def enroll_encoder_cell_x_embeddings(x, is_batch=False, is_serving=False, is_short=False):
        #string sentences to embeddings
        x_word_embeddings = sentence_to_embeddings(x, is_batch=is_batch, is_serving=is_serving, is_short=is_short)

        #get batch size
        batch_size = get_batch_size(is_batch, is_serving)

        #initialize encoder cell state variables, considering batch size
        encoder_cell.init_state_variables(batch_size = batch_size)

        #decide if short or long enrolling
        if is_short:
            max_sentence_length = max_sentence_length_short
        else:
            max_sentence_length = max_sentence_length_long
        
        #encode words
        for index in range(max_sentence_length):
            encoder_cell.run(x_word_embeddings[index], is_batch=is_batch)  

        #get the normalized code from encoder cell
        state = encoder_cell.get_state()

        return state, x_word_embeddings

    #enroll encoder, get code
    def enroll_encoder_cell(x, is_batch=False, is_serving=False, is_short=False):
        state, x_word_embeddings = enroll_encoder_cell_x_embeddings(x, is_batch=is_batch, is_serving=is_serving, is_short=is_short)

        #return only the state
        return state
        
    #enroll decoder, get list of decoded output embeddings
    def enroll_decoder_cell(x, is_batch=False, is_serving=False, is_short=False):
        #get batch size
        batch_size = get_batch_size(is_batch, is_serving)

        #initialize decoder cell state variables, considering batch size
        decoder_cell.init_state_variables(batch_size = batch_size)

        #decide if short or long enrolling
        if is_short:
            max_sentence_length = max_sentence_length_short
        else:
            max_sentence_length = max_sentence_length_long

        #decode words
        y = list() #list of output embeddings
        for index in range(max_sentence_length):
            if index == 0:
                #for the first enrolling, set the initial state as the input of the decoder cell
                decoder_cell.set_state(x)
            decoder_cell.run(decoder_cell.output, is_batch = is_batch)
            #store the outputs to a list
            y.append(decoder_cell.output)

        return y
        
    def inference_class(x, is_batch=False, is_serving=False, is_short=False):
        #encode with LSTM cell
        state = enroll_encoder_cell(x, is_batch=is_batch, is_serving=is_serving, is_short=is_short)

        #encode with fully connected layers
        code = autoencoder.run_encode(state)
        
        #logits
        logits = tf.nn.xw_plus_b(tf.nn.dropout(code, keep_prob = keep_prob_class), w_class, b_class)

        return logits

    def inference_autoencoder(x, is_batch=False, is_serving=False, is_short=False):
        #encode with LSTM cell
        state, x_word_embeddings = enroll_encoder_cell_x_embeddings(x, is_batch=is_batch, is_serving=is_serving, is_short=is_short)

        #encode and decode with fully connected layers
        code_decoded, code = autoencoder.run(state)

        #decode with LSTM cell
        y = enroll_decoder_cell(code_decoded, is_batch=is_batch, is_serving=is_serving, is_short=is_short)

        return y, x_word_embeddings, code


    #classification#
    ################

    #inference for classification
    logits_batch = inference_class(x_train_batch_class, is_batch=True, is_short=False)
    logits_training = inference_class(x_training_class, is_batch=False, is_short=False)

    #softmax layer
    probabilities_training = tf.nn.softmax(logits_training)

    #loss function for classifier (cross entropy)
    if is_multilabel:
        #if multiple labels can be true simultaneously
        loss_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train_batch_class_, logits=logits_batch))
    else:
        #if only one label can be true at once
        loss_class = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_train_batch_class_, logits=logits_batch))
    #add to tensorboard summary
    tf.summary.scalar('loss_class', loss_class)

    #optimizer for classification
    train_step_class = tf.train.AdamOptimizer(learning_rate_class).minimize(loss_class)

    #define accuracy op for train data
    def correct_prediction(x, y_):
        #FOR SINGLE LABEL ONLY
        logits = inference_class(x, is_batch=False, is_short=False)
        probabilities = tf.nn.softmax(logits)
        correct_prediction = tf.equal(
            tf.argmax(probabilities, axis=1),
            tf.argmax(y_, axis=0))
        return tf.cast(correct_prediction, tf.float32)
        
    correct_prediction_training = correct_prediction(x_training_class, y_training_class_)
    correct_prediction_test = correct_prediction(x_test_class, y_test_class_)

    #serving inference for classification
    logits_serving = inference_class(x_serving, is_batch=True, is_serving=True, is_short=False)
    probabilities_serving = tf.nn.softmax(logits_serving)
    y_serving = tf.argmax(probabilities_serving, axis=1)

    #autoencoder#
    #############

    #inference for autoencoder
    y_autoencoder, x_autoencoder_embedded, _ = inference_autoencoder(x_train_batch_autoencoder, is_batch=True, is_serving=False, is_short=True)

    #list of tensors -> one tensor with shape = (batch_size * max_sentence_length, embedding_size)
    y_autoencoder_concat = tf.concat(y_autoencoder, axis = 0)
    
    #calculate loss as the negative average cosine distance of the predicted and training words
    #reshape to a vector, because average cosine distance is a dot product
    y_autoencoder_concat_reshaped = tf.reshape(y_autoencoder_concat, [-1])
    x_autoencoder_embedded_reshaped = tf.reshape(x_autoencoder_embedded, [-1])
    #loss
    loss_autoencoder = -tf.reduce_mean(tf.multiply(y_autoencoder_concat_reshaped, x_autoencoder_embedded_reshaped)) * embedding_size
    #add to tensorboard summary
    tf.summary.scalar('loss_autoencoder', loss_autoencoder)

    #optimizer for autoencoder
    if optimizer == 'AdamOptimizer':
        train_step_autoencoder = tf.train.AdamOptimizer(learning_rate_autoencoder).minimize(loss_autoencoder)
    elif optimizer == 'AdadeltaOptimizer':
        train_step_autoencoder = tf.train.AdadeltaOptimizer(learning_rate_autoencoder).minimize(loss_autoencoder)
    elif optimizer == 'GradientDescentOptimizer':
        train_step_autoencoder = tf.train.GradientDescentOptimizer(learning_rate_autoencoder).minimize(loss_autoencoder)
    else:
        raise NameError('unkown optimizer')
    
    #serving inference for autoencoder
    y_autoencoder_serving, _, code_serving = inference_autoencoder(x_serving, is_batch=True, is_serving=True, is_short=True) 


    ###########
    #savemodel#
    ###########

    #save model for serving in google cloud
    def save_model(sess, x_serving, y_serving):
        if not is_autoencoder:
            #signatures
            model_signature = signature_def_utils.build_signature_def(
                      inputs={
                          "features": utils.build_tensor_info(x_serving)
                      },
                      outputs={
                          "prediction": utils.build_tensor_info(y_serving)
                      },
                      method_name=signature_constants.PREDICT_METHOD_NAME)

            #build
            export_path = './export/classifier'
                
            builder = saved_model_builder.SavedModelBuilder(export_path)
            builder.add_meta_graph_and_variables(
                        sess,
                        [tag_constants.SERVING],
                        signature_def_map={
                            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            model_signature,
                        },
                        #legacy_init_op=legacy_init_op)
                        legacy_init_op=tf.group(tf.initialize_all_tables(),
                                                name="legacy_init_op"))

            builder.save()
        else:
            print('autoencoder is not prepared for serving yet')


    ##########
    #training#
    ##########

    #initialization op
    init = tf.global_variables_initializer()

    #add ops to save and restore all the variables to checkpoint
    saver = tf.train.Saver()

    #accuracies to log
    accuracies_training = list()
    accuracies_test = list()

    #define training function
    def train():

        start_time = time.time()

        with tf.Session() as sess:

            #init session
            sess.run(init)
            tf.tables_initializer().run()
            #init for que reader
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            #load autoencoder checkpoint if transfer learning for classifier
            if is_transfer_learning_class and not is_autoencoder:
                export_path = './checkpoint/autoencoder/model'
                #restore model from checkpoint
                saver.restore(sess, export_path)
                print('Model restored for transfer learning')

            #setup tensorboard summaray
            if is_tensorboard_log:
                merged_summary = tf.summary.merge_all()
                tensorboard_writer = tf.summary.FileWriter('./log-directory')
                #tensorboard_writer.add_graph(sess.graph)

            #train
            for i in range(step_number):

                #classification#
                ################
                if not is_autoencoder:

                    #training step
                    _ = sess.run(train_step_class)

                    #print percentage and results
                    if (i % log_step_number_class) == 0:
                        #loss of the actual training sample
                        loss_class_out = sess.run(loss_class)

                        print(str((float(i) / step_number)*100) + '%        ' + str(loss_class_out))

                        logits_training_out, probabilities_training_out, x_training_class_out, y_training_class_out_ = sess.run([logits_training, probabilities_training, x_training_class, y_training_class_])
                        print('input training sentence:')
                        print(x_training_class_out)
                        print('input training class:')
                        print(y_training_class_out_)
                        print('model estimated probabilities:')
                        print(probabilities_training_out)

                        #tensorboard summary
                        if is_tensorboard_log:
                            merged_summary_out = sess.run(merged_summary)
                            tensorboard_writer.add_summary(merged_summary_out, i)

                    #print accuraccy
                    if (i % accuracy_step_number_class) == 0:
                        #training accuraccy                
                        correct_prediction_sum = 0
                        for _ in range(accuracy_sample_number_training):
                            correct_prediction_sum += sess.run(correct_prediction_training)
                        accuracy_training = correct_prediction_sum / float(accuracy_sample_number_training)
                        accuracies_training.append(accuracy_training)
                        print('accuracy training:')
                        print(accuracy_training)
                        
                        #test accuraccy
                        correct_prediction_sum = 0
                        for _ in range(accuracy_sample_number_test):
                            correct_prediction_sum += sess.run(correct_prediction_test)
                        accuracy_test = correct_prediction_sum / float(accuracy_sample_number_test)
                        accuracies_test.append(accuracy_test)
                        print('accuracy test:')
                        print(accuracy_test)

                #autoencoder#
                #############
                else:

                    #train step
                    _ = sess.run(train_step_autoencoder)

                    #print prcentage and results
                    if (i % log_step_number_autoencoder) == 0:
                        #losses
                        loss_autoencoder_out = sess.run(loss_autoencoder)

                        print(str((float(i) / step_number)*100) + '%        ' + str(loss_autoencoder_out))

                        #print an encoded and decoded sentence
                        y_autoencoder_concat_out, x_autoencoder_embedded_out, x_train_batch_autoencoder_out = sess.run([y_autoencoder_concat, x_autoencoder_embedded, x_train_batch_autoencoder])
                        #reshape to get the first sentences
                        x_train_batch_autoencoder_first = x_train_batch_autoencoder_out[0]
                        x_autoencoder_embedded_first = x_autoencoder_embedded_out[:,0,:]
                        y_autoencoder_concat_first = y_autoencoder_concat_out[0::batch_size_training]
                        
                        #print
                        print('training sentence, training sentence embedded, model estimation:')
                        print(x_train_batch_autoencoder_first)
                        print(sentence_embedded_to_string(embeddings, x_autoencoder_embedded_first))
                        print(sentence_embedded_to_string(embeddings, y_autoencoder_concat_first))                 

                        #tensorboard summary
                        if is_tensorboard_log:
                            merged_summary_out = sess.run(merged_summary)
                            tensorboard_writer.add_summary(merged_summary_out, i)


            training_time = time.time() - start_time
            print('Training time: ' + str(training_time))

            #save model
            if is_save_model:
                save_model(sess, x_serving, y_serving)

            #save model checkpoint
            if is_save_model_checkpoint:
                if not is_autoencoder:
                    export_path = './checkpoint/classifier/model'
                else:
                    export_path = './checkpoint/autoencoder/model'
                save_path = saver.save(sess, export_path)
                print('Model saved in file: ' + save_path)

            #stop que reader
            coord.request_stop()
            coord.join(threads)


    ###################
    #use trained model#
    ###################

    #apply model to all gigers news
    def process_gigers_news(sess, is_use_autoencoder=False):
        #load (ids, titles) as list
        ids, titles = data_loader.gigers_data_list()
        
        #classes and codes (the output of the model) one or the other will be used
        if not is_use_autoencoder:
            classes = list()
        else:
            codes = list()

        #loop through all titles, in batch_size_serving sized batches
        x_serving_feed_list = list()
        title_counter = 0
        for index, title in enumerate(titles):
            x_serving_feed_list.append(title)
            title_counter = title_counter + 1

            #if last incomplete batch, pad batch
            if index == len(titles) - 1:
                while len(x_serving_feed_list) < batch_size_serving:
                    x_serving_feed_list.append(title)
                title_counter = batch_size_serving

            ##classify with model
            if title_counter == batch_size_serving:
                #pass titles to feed_dict
                feed_dict = dict()
                feed_dict[x_serving] = x_serving_feed_list

                if not is_use_autoencoder:
                    #classify with model
                    y_serving_out = sess.run(y_serving, feed_dict=feed_dict)
                    classes.extend(y_serving_out)
                else:
                    #encode with the model
                    code_serving_out = sess.run(code_serving, feed_dict=feed_dict)
                    codes.extend(code_serving_out)

                #print status
                print(str(index) + ' / ' + str(len(titles)))

                #start new batch
                title_counter = 0
                x_serving_feed_list = list()

        if not is_use_autoencoder:
            #cap classes to the numer of titles (necessarry because of the last padded batch)
            classes = classes[:len(titles)]
            #write to csv file
            data_loader.write_gigers_data_classified(ids, titles, classes)
        else:
            #cap codes to the numer of titles (necessarry because of the last padded batch)
            codes = codes[:len(titles)]
            #write to csv file
            data_loader.write_gigers_data_encoded(ids, titles, codes)

    #define function to search closest codes to the imput code, and output them as string
    def search_closest(encodings_list, encoding, sentence_list_string, top_k = 8):
        #top_k: number of nearest codes
        dist = np.dot(encodings_list, np.transpose(encoding))
        nearest_indeces = (-dist).argsort(axis=0)[0:top_k-1]
        search_results = [sentence_list_string[index] for index in nearest_indeces]
        return search_results   

    #search the embedding in all the embedding of gigers news
    def search_news_in_all_encoded(code_search):
        #read all encoded news
        ids, titles, codes = data_loader.read_gigers_data_encoded()
        
        #search closest senteces to code_search
        search_results = search_closest(codes, code_search, titles, top_k = 18)

        print('search results:')
        for search_result in search_results:
            print(search_result)

    #apply the model on a single new
    def process_single_news(sess, is_use_autoencoder=False, is_search_news_in_all_encoded=False):

        sentence_string = 'Stream Steve Rachmads album as Sterac Electronics'
        sentence_length = max_sentence_length_short
        

        #fill up serving input with single news
        feed_dict = dict()
        x_serving_feed_list = list()
        for _ in range(batch_size_serving):
            x_serving_feed_list.append(sentence_string)
        feed_dict[x_serving] = x_serving_feed_list

        if not is_use_autoencoder:
            #inference from model
            y_serving_out, probabilities_serving_out = sess.run([y_serving, probabilities_serving], feed_dict=feed_dict)

            #print
            print('y_serving_out')
            print(y_serving_out)
            print('probabilities_serving_out')
            print(probabilities_serving_out)
        else:
            #inference from model
            y_autoencoder_serving_out, code_serving_out = sess.run([y_autoencoder_serving, code_serving], feed_dict=feed_dict)
            print('y_autoencoder_serving_out')

            #reshape y_autoencoder_serving to sentence
            sentence_serving = np.empty([sentence_length, embedding_size])
            #shape of y_autoencoder_serving: [[sentence_1_word_1, sentence_2_word_1, ...], [sentence_1_word_2, sentence_2_word_2, ...] ...
            for index, word_batch_serving in enumerate(y_autoencoder_serving_out):
                sentence_serving[index] = word_batch_serving[0]

            #print
            print(sentence_embedded_to_string(embeddings, sentence_serving))
            print('code_serving_out')

            #search
            if is_search_news_in_all_encoded:
                search_news_in_all_encoded(code_serving_out[0])
            #print(code_serving_out[0])

    #restore model from checkpoint and use the model
    def load_model_checkpoint_and_process():
        process_type = 'EncodeSingleNews'

        if not is_autoencoder:
            export_path = './checkpoint/classifier/model'
        else:
            export_path = './checkpoint/autoencoder/model'        

        with tf.Session() as sess:
            #sess.run(init)
            tf.tables_initializer().run()

            #restore model from checkpoint
            saver.restore(sess, export_path)
            print('Model restored')

            #use the model
            if process_type == 'ClassifyGigersNews':
                process_gigers_news(sess, is_use_autoencoder=False)
            elif process_type == 'EncodeGigersNews':
                process_gigers_news(sess, is_use_autoencoder=True)
            elif process_type == 'ClassifySingleNews':
                process_single_news(sess, is_use_autoencoder=False)
            elif process_type == 'EncodeSingleNews':
                process_single_news(sess, is_use_autoencoder=True, is_search_news_in_all_encoded=False)
            elif process_type == 'SearchSingleNews':
                process_single_news(sess, is_use_autoencoder=True, is_search_news_in_all_encoded=True)

            #save model
            if is_save_model:
                save_model(sess, x_serving, y_serving)


    ################
    #function calls#
    ################

    #if not loading model then train
    if is_train_model:
        #call training function
        train()
    else:
        #load model
        load_model_checkpoint_and_process()


######
#main#
######
if __name__ == '__main__':
    trainer_run()
