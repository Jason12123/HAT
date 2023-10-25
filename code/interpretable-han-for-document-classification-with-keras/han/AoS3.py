
from tensorflow import keras 
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input, Embedding, Dense, GRU, Bidirectional, TimeDistributed, Lambda, Dropout, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
import numpy as np
import pickle
import os
from glob import glob

from .attention import Attention

DENSE_SIZE = 100
Special_value= -10


class HAN(Model):
    def __init__(self, embedding_matrix, max_sent_length=100,
                 max_sent_num=25, max_doc_num=5, word_embed_dim=100, sent_embed_dim=100, doc_embed_dim=5):
        """Implementation of Hierarchical Attention Networks for document classification.

        Args:
            embedding_matrix: embedding matrix used to represent words.
            max_sent_length: int, maximum number of words per sentence, default is 100.
            max_sent_num: int, maximum number of sentences accepted, default is 10.
            word_embed_dim: int, dimension of word encoder, default is 100.
            sent_embed_dim: int, dimension of sentence encoder, default is 100.
        
            ************************************
            max_doc_num: int, maximum number of documents per patient, default is 5.
            doc_embed_dim: int, dimension of document encoder, default is 300.  
            
        """
        self.embedding_matrix = embedding_matrix
        self.max_sent_length = max_sent_length
        self.max_sent_num = max_sent_num
        self.word_embed_dim = word_embed_dim
        self.sent_embed_dim = sent_embed_dim
        self.max_doc_num=max_doc_num
        self.doc_embed_dim=doc_embed_dim

        super(HAN, self).__init__(name='han')
        self.build_model()

    def build_word_encoder(self):
        """Build word encoder.

        The function starts with a Input tensor layer, and go through
        Embedding layer and then Bidirectional GRU layer and
        TimeDistributed layer and ends with Attention.

        Returns:
            Model, a model layer wraps sent_input and word attention.
        """
        sent_input = Input(shape=(self.max_sent_length,), dtype='float32')
        
        #masked1=Masking(mask_value=Special_value)(sent_input)
        embedded_sent = Embedding(
            self.embedding_matrix.shape[0], self.embedding_matrix.shape[1],
            weights=[self.embedding_matrix], input_length=self.max_sent_length,
            trainable=True
        )(sent_input)
        
        #Masking
        
        
        # For Bidirectional, devide by 2
        encoded_sent = Bidirectional(
            GRU(int(self.word_embed_dim / 2), return_sequences=True)
        )(embedded_sent)
        # TODO: check if dense is still needed in timedistributed
        selfattention_word=SeqSelfAttention(attention_activation='sigmoid',attention_width=5)(encoded_sent)
        input_merged_1 = Concatenate()([embedded_sent,selfattention_word])


        dense_sent = TimeDistributed(Dense(DENSE_SIZE))(input_merged_1)
        word_att = Attention(name='word_attention')(dense_sent)

        return Model(sent_input, word_att)

    def build_sent_encoder(self):
        """Build sentence encoder.

        Perform a Bidirectional GRU layer, and then a TimeDistributed
         Dense layer before going to the Attention.

        Args:
            sent_encoder: the input sentence encoder.

        Returns:
            doc_att: sentence attention weights.
        """
        
        text_input = Input(shape=(self.max_sent_num, self.max_sent_length,))
        
        #masked2=Masking(mask_value=Special_value)(text_input)
        # encode sentences into a single vector per sentence
        self.model_word = self.build_word_encoder()
        # time distribute word model to accept text input
        sent_encoder = TimeDistributed(self.model_word)(text_input)
        
        
        # For Bidirectional, devide by 2
        encoded_text = Bidirectional(
            GRU(int(self.sent_embed_dim / 2), return_sequences=True)
        )(sent_encoder)

        selfattention_sent=SeqSelfAttention(attention_activation='sigmoid',attention_width=5)(encoded_text)
        input_merged_2 = Concatenate()([sent_encoder,selfattention_sent])
        dense_text = TimeDistributed(Dense(DENSE_SIZE))(input_merged_2)
        doc_att = Attention(name='sent_attention')(dense_text)
        return Model(text_input, doc_att)
    
    def build_doc_encoder(self, doc_encoder):
        
        encoded_doc = Bidirectional(
            GRU(int(self.doc_embed_dim / 2), return_sequences=True)
        )(doc_encoder)

        selfattention_doc=SeqSelfAttention(attention_activation='sigmoid',attention_width=5)(encoded_doc)
        input_merged_3 = Concatenate()([doc_encoder,selfattention_doc])
        dense_doc = TimeDistributed(Dense(DENSE_SIZE))(input_merged_3)
        document_att = Attention(name='doc_attention')(dense_doc)
        return document_att

    def build_model(self):
        """Build the embed and encode models for word and sentence.

        For the word model, the sequence of Layers is: Embedding ->
        Bidirectional(GRU) -> TimeDistributed(Dense) -> Attention

        For the sentence model, it takes the word level model as input
        for TimeDistributed Layer to make sentence encoder. And the
        sequence is: TimeDistributed(WordModel) -> Bidirectional(GRU)
        -> TimeDistributed(Dense) -> Attention -> Dense

        There is no output, but will save the word and sentence models.
        
        text_input = Input(shape=(self.max_sent_num, self.max_sent_length))
        # encode sentences into a single vector per sentence
        self.model_word = self.build_word_encoder()
        # time distribute word model to accept text input
        sent_encoder = TimeDistributed(self.model_word)(text_input)

        doc_att = self.build_sent_encoder(sent_encoder)
        
        
        """
        doc_input = Input(shape=(self.max_doc_num, self.max_sent_num, self.max_sent_length))
        
        #masked3=Masking(mask_value=Special_value)(doc_input)
        
        self.model_sent = self.build_sent_encoder()
        
        doc_encoder= TimeDistributed(self.model_sent)(doc_input)
        
        document_att= self.build_doc_encoder(doc_encoder)
        
        #doc_att = self.build_sent_encoder(sent_encoder)
        # dense the output to 2 because the result is a binary classification.
        output_tensor = Dense(2, activation='softmax', name='classification')(document_att)
        # Create Sentence-level Model
        self.model = Model(doc_input, output_tensor)

    def print_summary(self):
        """Print the model summary for both word and sentence level model.
        """
        print("Word Level")
        self.model_word.summary()
        
        print("Sent Level")
        self.model_sent.summary()

        print("Doc Level")
        self.model.summary()

    def train_model(self, checkpoint_path, X_train, y_train, X_val, y_val, X_test, y_test,
                    optimizer='adagrad', loss='categorical_crossentropy',
                    metric='acc', monitor='val_loss', batch_size=20, epochs=10):
        """Train the HAN model.

        Args:
            checkpoint_path: str, the path to save checkpoint file.
            X_train: training dataset.
            y_train: target of training dataset.
            X_test: testing dataset.
            y_test: target of testing dataset.
            optimizer: optimizer for compiling, default is adagrad.
            loss: loss function, default is categorical_crossentropy.
            metric: measurement metric, default is acc (accuracy).
            monitor: monitor of metric to pick up best weights, default is val_loss.
            batch_size: batch size, default is 20.
            epochs: number of epoch, default is 10.
        """
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[metric]
        )
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            verbose=1, save_best_only=True
        )

        self.model.fit(
            X_train, y_train, batch_size=batch_size, epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint]
        )

        results=self.model.evaluate(X_test, y_test, batch_size=10)
        print('test loss, test acc:', results)
 #       yhat = self.model.predict(X_test)
 #       yhat_classes=list(map(lambda x: np.argmax(x), yhat))
        # reduce to 1d array
 #       yhat_probs = yhat[:, 1]
 
 
        # accuracy: (tp + tn) / (p + n)
        #accuracy = accuracy_score(list(map(lambda x: np.argmax(x), yv)), list(map(lambda x: np.argmax(x), yhat)))
        #print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
 #       precision = precision_score(y_test[:,1], yhat_classes)
        #print('Precision: %f' % precision)
 #       # recall: tp / (tp + fn)
 #       recall = recall_score(y_test[:,1], yhat_classes)
        #print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
 #       f1 = f1_score(y_test[:,1], yhat_classes)
 #       print('F1 score: %f' % f1)
 
        # kappa
  #      kappa = cohen_kappa_score(y_test[:,1], yhat_classes)
        #print('Cohens kappa: %f' % kappa)
        # ROC AUC
 #       auc = roc_auc_score(y_test[:,1], yhat_probs)
        #print('ROC AUC: %f' % auc)
        # confusion matrix
 #       matrix = confusion_matrix(y_test[:,1], yhat_classes)
 #       print('test loss, test acc:', results)
        return results
        

    def show_word_attention(self, x):
        """Show the prediction of the word level attention.

        Args:
            x: the input array with size of (max_sent_length,).

        Returns:
            Attention weights.
        """
        att_layer = self.model_word.get_layer('word_attention')
        prev_tensor = att_layer.input

        # Create a temporary dummy layer to hold the
        # attention weights tensor
        dummy_layer = Lambda(
            lambda x: att_layer._get_attention_weights(x)
        )(prev_tensor)

        return Model(self.model_word.input, dummy_layer).predict(x)

    def show_sent_attention(self, x):
        """Show the prediction of the sentence level attention.

        Args:
            x: the input array with the size of (max_sent_num, max_sent_length).

        Returns:
            Attention weights.
        """
        att_layer = self.model_sent.get_layer('sent_attention')
        prev_tensor = att_layer.input

        dummy_layer = Lambda(
            lambda x: att_layer._get_attention_weights(x)
        )(prev_tensor)

        return Model(self.model_sent.input, dummy_layer).predict(x)
    
    def show_doc_attention(self, x):
        """Show the prediction of the sentence level attention.

        Args:
            x: the input array with the size of (max_sent_num, max_sent_length).

        Returns:
            Attention weights.
        """
        att_layer = self.model.get_layer('doc_attention')
        prev_tensor = att_layer.input

        dummy_layer = Lambda(
            lambda x: att_layer._get_attention_weights(x)
        )(prev_tensor)

        return Model(self.model.input, dummy_layer).predict(x)

    @staticmethod
    def word_att_to_df(sent_tokenized_review, word_att):
        """Convert the word attention arrays into pandas dataframe.

        Args:
            sent_tokenized_review: sentence tokenized review, which means sent_tokenize(review)
                has to be executed beforehand. And only one review is allowed, since it's
                on word attention level, and also it's the required input size in
                self.show_word_attention, but review can contain multiple sentences.
            word_att: attention weights obtained from self.show_word_attention.

        Returns:
            df: pandas.DataFrame, contains original reviews column and word_att column,
                and word_att column is a list of dictionaries in which word as key while
                corresponding weight as value.
        """
        # remove the trailing dot
        ori_sents = [i.rstrip('.') for i in sent_tokenized_review]
        # split sentences into words
        ori_words = [x.split() for x in ori_sents]
        # truncate attentions to have equal size of number of words per sentence
        truncated_att = [i[-1 * len(k):] for i, k in zip(word_att, ori_words)]

        # create word attetion pair as dictionary
        word_att_pair = []
        for i, j in zip(truncated_att, ori_words):
            word_att_pair.append(dict(zip(j, i)))

        return pd.DataFrame([(x, y) for x, y in zip(word_att_pair, ori_words)],
                            columns=['word_att', 'transcript'])

    @staticmethod
    def sent_att_to_df(sent_tokenized_reviews, sent_att):
        """Convert the sentence attention arrays into pandas dataframe.

        Args:
            sent_tokenized_reviews: sent tokenized reviews, if original input is a Series,
                that means at least Series.apply(lambda x: sent_tokenize(x)) has to be
                executed beforehand.
            sent_att: sentence attention weight obtained from self.show_sent_attetion.

        Returns:
            df: pandas.DataFrame, contains original reviews column and sent_att column,
                and sent_att column is a list of dictionaries in which sentence as key
                while corresponding weight as value.
        """
        # create reviews attention pair list
        reviews_atts = []
        for review, atts in zip(sent_tokenized_reviews, sent_att):
            review_list = []
            for sent, att in zip(review, atts):
                # each is a list of dictionaries
                review_list.append({sent: att})
            reviews_atts.append(review_list)
        return pd.DataFrame([(x, y) for x, y in zip(reviews_atts, sent_tokenized_reviews)],
                            columns=["sent_att", "transcript"])
    
    
    
    def doc_att_to_df(reviews, doc_att):
        docs_atts = []
        for review, atts in zip(reviews, doc_att):
            doc_list = []
            for doc, att in zip(review, atts):
                doc_list.append({doc: att})
            docs_atts.append(doc_list)
                
        
        return pd.DataFrame([(x, y) for x, y in zip(docs_atts,reviews)],
                            columns=["doc_att", "transcript"])