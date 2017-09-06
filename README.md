# gigers_lstm
This project aims to apply unsupervised and supervised learning methods on text data and utilise the resulting model for some useful applications:

* classification after an additional supervised training

* robust searching in the text

## Classifier

First the sentence is encoded with an LSTM encoding RNN, then the state of the RNN is fed through a deep autoencoder (this is optional), then the resulting state is given to a classifier layer (with softmax), which calculates the estimated class.

## Autoencoder

First the sentence is encoded with an LSTM encoding RNN, then the state of the RNN is fed through a deep autoencoder (this is optional), then the resulting state is given to an LSTM decoding RNN which tries to reconstruct the input sentence. After training the model is capable of transforming a given sentence into a dense code, which can be used for the mentioned applications.
