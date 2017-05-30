# -*- coding: utf-8 -*-
"""this module is implementation of matching question and answer using CNN to
   compute the cosine distance between question and answer"""

from keras.layers import Input, Conv1D, Dense, MaxPooling1D, Bidirectional, LSTM, Dropout, TimeDistributed, Multiply, Flatten
from keras.layers.core import Lambda
from keras.models import Model
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from gensim.models import Word2Vec
import numpy as np
from evaluation import raw_result_parser
from evaluation import tagging_util

def find_index(list, word):
    try:
        return list.index(word)
    except:
        return -1

class QACOSModel3:
    def __init__(self):
        self.wordModel = Word2Vec.load('model_new_and_wiki')
        self.m = 0.009
        self.model = None
        self.word_dim = 100
        self.question_len = 30
        self.evidence_len = 80
        self.r_dim = 100

    def word_vec(self, word):
        try:
            return self.wordModel[word]
        except KeyError:
            return np.zeros((self.word_dim,))

    def get_model(self):
        dropout_rate = 0.05
        D = self.r_dim
        margin = 0.1
        q_input = Input(shape=(self.question_len, self.word_dim), name='question')
        q = Bidirectional(LSTM(D,
                           dropout=dropout_rate,
                           return_sequences=True,
                           name='q'))(q_input)

        q_alpha_tanh = TimeDistributed(Dense(D, activation='tanh'))(q)
        q_alpha = TimeDistributed(
            Dense(1, activation='softmax', name='q_alpha')
        )(q_alpha_tanh)
        q_alpha_multiply = Multiply()([q, q_alpha])
        question_represent = Lambda(lambda x: K.sum(x, axis=1), name='question_representation')(q_alpha_multiply)

        e_input = Input(shape=(self.evidence_len, self.word_dim), name='answer')
        e = Bidirectional(LSTM(D,
                               dropout=dropout_rate,
                               return_sequences=True,
                               name='q'))(e_input)

        a_alpha_tanh = TimeDistributed(Dense(D, activation='tanh'))(e)
        a_alpha = TimeDistributed(
            Dense(1, activation='softmax', name='a_alpha')
        )(a_alpha_tanh)
        a_alpha_multiply = Multiply()([e, a_alpha])
        answer_represent = Lambda(lambda x: K.sum(x, axis=1), name='answer_representation')(a_alpha_multiply)

        '''compute the cosine of r_q and r_a as the concatenate input of r_q and r_a'''

        def cosine_distance(vests):
            x, y = vests
            x = K.l2_normalize(x, axis=-1)
            y = K.l2_normalize(y, axis=-1)
            return -K.mean(x * y, axis=-1, keepdims=True)

        def cos_dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)

        def myloss(ypred, ytrue):
            return K.maximum(0, margin - ypred[0] + ypred[1])


        cos_output = Lambda(function=cosine_distance, output_shape=cos_dist_output_shape)([question_represent, answer_represent])
        model = Model(inputs=[q_input, e_input], outputs=[cos_output])
        model.summary()
        model.compile(optimizer='rmsprop', loss=myloss, loss_weights=[0.01])
        self.model = model
        return self.model

    def load_train_data(self, data_dir):
        data = raw_result_parser.load_train_data(data_dir)
        load_max = 100
        load_cur = 0
        for q_token, e_token, golden_label, g1, g2 in data:
            output_l = 0
            input_q, input_e = self.get_input_q_e(q_token, e_token)
            begin = find_index(golden_label, [1, 0, 0, 0])
            inside = find_index(golden_label, [0, 1, 0, 0])
            if begin != -1 or inside != -1:
                output_l = 1
            yield input_q, input_e, output_l
            load_cur += 1
            if load_cur > load_max:
                break

    ''' @:param the list of split question and evidence
        @:return a maxtric which col is the word_vec of the split word'''
    def get_input_q_e(self, question, evidence):
        input_q = np.zeros((self.question_len, self.word_dim))
        input_e = np.ones((self.evidence_len, self.word_dim))
        for word_index in range(min(self.question_len, len(question))):
            word = question[word_index]
            input_q[word_index] = self.word_vec(word)
        for word_index in range(min(self.evidence_len, len(evidence))):
            word = evidence[word_index]
            input_e[word_index] = self.word_vec(word)
        return input_q, input_e

    def fit(self, input_qs, input_es, output_ls):
        self.model.fit([input_qs, input_es], [output_ls], epochs=10, batch_size=60)

    def save_model(self, model_file):
        self.model.save_weights(model_file)

    def load_model(self, model_file):
        self.get_model()
        self.model.load_weights(model_file)
        return self.model

    ''':param input_qes and input_ees should be type() as np.array'''
    def predict(self, input_qs, input_es):
        if type(input_qs) is list:
            input_qs = np.array(input_qs)
        if type(input_es) is list:
            input_es = np.array(input_es)
        return self.model.predict([input_qs, input_es])


if __name__=='__main__':
    qacos = QACOSModel3()
    qacos.get_model()
    iters = qacos.load_train_data('WebQA.v1.0/data/training.json.gz')
    train_qs = []
    train_es = []
    train_labels = []
    for train_q, train_e, train_label in iters:
        train_qs.append(train_q)
        train_es.append(train_e)
        train_labels.append(train_label)

    train_qs = np.array(train_qs, dtype=np.float32)
    train_es = np.array(train_es, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)
    qacos.fit(train_qs, train_es, train_labels)
    qacos.save_model('qacosmodel_0.3.h5')
