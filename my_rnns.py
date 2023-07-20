import os
import requests
import tensorflow as tf
import numpy as np 
import time
from matplotlib import pyplot as plt


class process_data():
    def __init__(self,text):
        self.text = text
        
    def get_uniq_char(self):
        vocab = sorted(set(self.text))
        self.vocab = vocab
        return vocab
    
    def tokenize(self):
        tokenize = tf.keras.layers.StringLookup(vocabulary=self.vocab)
        self.tokenize = tokenize
        return tokenize
    
    def char_from_ids(self):
        tokenizer = self.tokenize
        chars_from_ids = tf.keras.layers.StringLookup(
                         vocabulary=tokenizer.get_vocabulary(), invert=True, mask_token=None )
        return chars_from_ids
    
    def text_from_ids(self,ids):
        tokenizer = self.tokenize
        chars_from_ids = tf.keras.layers.StringLookup(
                         vocabulary=tokenizer.get_vocabulary(), invert=True, mask_token=None )
        return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
    
    
class data_set():
    def __init__(self,text,seq_length,batch_size,split):
        self.text = text
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.split = split
    
    def split_input_target(sequence):
            input_text = sequence[:-1]
            target_text = sequence[1:]
            return input_text, target_text
        
    def prepare(self,tokenizer):
            all_ids = tokenizer(tf.strings.unicode_split(self.text, 'UTF-8'))
            ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

            sequences = ids_dataset.batch(self.seq_length+1, drop_remainder=True)

            dataset = sequences.map(data_set.split_input_target)

            buffer_size = 10000

            dataset = (
                dataset
                .shuffle(buffer_size)
                .batch(self.batch_size, drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE))

            train_size = int(self.split * len(dataset))
            val_size = len(dataset) - train_size

            train_dataset = dataset.take(train_size)
            val_dataset = dataset.skip(train_size)

            return train_dataset,val_dataset
        
        
        
        
class initializers():
    def __init__(self,vocab_size):
        self.vocab_size = vocab_size
        self.embedding_dim = 256
        self.rnn_units = 1024
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = 'adam'
        self.metric = ['accuracy']
        self.epochs = 30
        self.callbacks=[
#             tf.keras.callbacks.ModelCheckpoint(filepath='my_model.h5',
#                                                 monitor='val_loss', 
#                                                 save_best_only=True, 
#                                                 mode='min', 
#                                                 verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             mode='auto',
                                             verbose=1,
                                             patience=5),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                patience=3,
                                                verbose=1,
                                                factor=.5, 
                                                min_lr=0.0000001)
        ]
        
        
class GRU_Model(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        x = initializers(vocab_size)
        self.embedding_dim = x.embedding_dim
        self.rnn_units = x.rnn_units
        self.loss = x.loss
        self.optimizer = x.optimizer
        self.metric = x.metric
        self.epochs = x.epochs
        self.callbacks = x.callbacks
        
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = tf.keras.layers.GRU(self.rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(self.vocab_size)
        
    def call(self, inputs, states=None, return_state=False, training=False):
            x = inputs
            x = self.embedding(x, training=training)
            if states is None:
                  states = self.gru.get_initial_state(x)
            x, states = self.gru(x, initial_state=states, training=training)
            x = self.dense(x, training=training)

            if return_state:
                  return x, states
            else:
                  return x
              
                
class LSTM_Model(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        x = initializers(vocab_size)
        self.embedding_dim = x.embedding_dim
        self.rnn_units = x.rnn_units
        self.loss = x.loss
        self.optimizer = x.optimizer
        self.metric = x.metric
        self.epochs = x.epochs
        self.callbacks = x.callbacks
        
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(self.vocab_size)
        
    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.lstm.get_initial_state(x)
        x, states_h, states_c = self.lstm(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, [states_h, states_c]
        else:
            return x
        
        
        
class RNN_Model(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        x = initializers(vocab_size)
        self.embedding_dim = x.embedding_dim
        self.rnn_units = x.rnn_units
        self.loss = x.loss
        self.optimizer = x.optimizer
        self.metric = x.metric
        self.epochs = x.epochs
        self.callbacks = x.callbacks
        
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = tf.keras.layers.SimpleRNN(self.rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(self.vocab_size)
        
    def call(self, inputs, states=None, return_state=False, training=False):
            x = inputs
            x = self.embedding(x, training=training)
            if states is None:
                  states = self.rnn.get_initial_state(x)
            x, states = self.rnn(x, initial_state=states, training=training)
            x = self.dense(x, training=training)

            if return_state:
                  return x, states
            else:
                  return x
              
                
              
class OneStep(tf.keras.Model):
       def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
            super().__init__()
            self.temperature = temperature
            self.model = model
            self.chars_from_ids = chars_from_ids
            self.ids_from_chars = ids_from_chars

            # Create a mask to prevent "[UNK]" from being generated.
            skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
            sparse_mask = tf.SparseTensor(
                # Put a -inf at each bad index.
                values=[-float('inf')]*len(skip_ids),
                indices=skip_ids,
                # Match the shape to the vocabulary
                dense_shape=[len(ids_from_chars.get_vocabulary())])
            self.prediction_mask = tf.sparse.to_dense(sparse_mask)

       @tf.function
       def generate_one_step(self, inputs, states=None):
                # Convert strings to token IDs.
                input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
                input_ids = self.ids_from_chars(input_chars).to_tensor()

                # Run the model.
                # predicted_logits.shape is [batch, char, next_char_logits]
                predicted_logits, states = self.model(inputs=input_ids, states=states,
                                                      return_state=True)
                # Only use the last prediction.
                predicted_logits = predicted_logits[:, -1, :]
                predicted_logits = predicted_logits/self.temperature
                # Apply the prediction mask: prevent "[UNK]" from being generated.
                predicted_logits = predicted_logits + self.prediction_mask

                # Sample the output logits to generate token IDs.
                predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
                predicted_ids = tf.squeeze(predicted_ids, axis=-1)

                # Convert from token ids to characters
                predicted_chars = self.chars_from_ids(predicted_ids)

                # Return the characters and model state.
                return predicted_chars, states
            
            
            
            
            
class word_level_process_data():
    def __init__(self,text):
        self.text = text
    def word_level_process(self):
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(self.text)    
        print(f'\n The total words are :{len(tokenizer.word_index)+1} ')
        input_ = input('\n\nDo u want to print the word dictionary? (y/n)\n\n')
        if input_ == 'y':
            print(f'\n\nThe word index dictionary is :\n\n {tokenizer.word_index}')
        return tokenizer
    
    
    
    
class data_set_word_level():
    def __init__(self,text,tokenizer):
        self.text = text
        self.tokenizer = tokenizer
    def gen_data(self):
        # Initialize the sequences list
        input_sequences = []

        # Loop over every line
        for line in self.text:

            # Tokenize the current line
            token_list = self.tokenizer.texts_to_sequences([line])[0]

            # Loop over the line several times to generate the subphrases
            for i in range(1, len(token_list)):

                # Generate the subphrase
                n_gram_sequence = token_list[:i+1]

                # Append the subphrase to the sequences list
                input_sequences.append(n_gram_sequence)

        # Get the length of the longest line
        max_sequence_len = max([len(x) for x in input_sequences])

        # Pad all sequences
        input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

        # Create inputs and label by splitting the last token in the subphrases
        X, labels = input_sequences[:,:-1],input_sequences[:,-1]

        # Convert the label into one-hot arrays
        Y = tf.keras.utils.to_categorical(labels, num_classes=len(self.tokenizer.word_index)+1)
        
        return X , Y , max_sequence_len