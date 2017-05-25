# -*- coding:utf-8 -*-
from DBBASEParser import DBBASEParser
from keras.layers import Input, LSTM, Dense, TimeDistributed, Lambda
from keras.models import Model
from gensim.models import Word2Vec
from keras.backend import tensorflow_backend as K
import numpy as np
from QAModel import QAModel
import jieba

class DBModel:
    def __init__(self):
        self.q_len = 30
        self.e_len = 100
        self.word_dim = 100
        self.qa_model = QAModel()
        self.qa_model.load_model('qamodel.h5')
        self.model = None

    def get_model(self):
        input_layer = Input(shape=(self.e_len, 4), name='e_layer_input')
        lstm_layer = LSTM(1, return_sequences=True)(input_layer)
        reshape_layer = Lambda(lambda x: K.reshape(x, [-1, self.word_dim]))(lstm_layer)
        output_layer = Dense(1, activation='sigmoid')(reshape_layer)
        self.model = Model(inputs=[input_layer], outputs=[output_layer])
        self.model.summary()
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[0.01])
        return self.model

    def load_data(self, db_file):
        parse = DBBASEParser(db_file)
        input_qs = []
        input_es = []
        labels = []
        max_iter = 100
        cur = 0
        for q, e, l in parse.iter_result():
            input_q, input_e = self.qa_model.get_input_q_and_e(q, e)
            input_qs.append(input_q)
            input_es.append(input_e)
            labels.append(l)
            if cur > max_iter:
                break
            else:
                cur += 1
        label_seq = self.qa_model.predict(input_qs, input_es)
        labels = np.array(labels)
        return label_seq, labels

    def fit(self, db_file):
        seq, labels = self.load_data(db_file)
        self.model.fit([seq], [labels],  epochs=10, batch_size=60)

    def save_model(self, model_file):
        self.model.save_weights(model_file)

    def load_model(self, model_file):
        self.get_model()
        self.model.load_weights(model_file)
        return self.model

    def predict(self, input_qs, input_es):
        if type(input_qs) is list:
            input_qs = np.array(input_qs)
        if type(input_es) is list:
            input_es = np.array(input_es)
        seq = self.qa_model.predict(input_qs, input_es)
        return self.model.predict([seq])

    def predict_by_sentences(self, sentence_qs, sentence_es):
        qs = []
        es = []
        for q_list, e_list in zip(sentence_qs, sentence_es):
            if type(q_list) is not list:
                q_list = jieba.cut(q_list)
            if type(e_list) is not list:
                e_list = jieba.cut(e_list)
            input_q, input_e = self.qa_model.get_input_q_and_e(q_list, e_list)
            qs.append(input_q)
            es.append(input_e)
        return self.predict(qs, es)

if __name__ == '__main__':
    db_file = 'BoP2017_DBAQ_dev_train_data/BoP2017-DBQA.train.txt'
    db_model = DBModel()
    db_model.get_model()
    db_model.fit(db_file)
    db_model.save_model('db_model.h5')
    db_test_file = 'BoP2017_DBAQ_dev_train_data/BoP2017-DBQA.dev.txt'
    test_parse = DBBASEParser(db_test_file)
    qs = []
    es = []
    ls = []
    max_ = 30
    cur = 0
    for q, e, l in test_parse.iter_result():
        qs.append(q)
        es.append(e)
        ls.append(l)
        if cur > max_:
            break
        else:
            cur += 1
    print(db_model.predict_by_sentences(qs, es))
