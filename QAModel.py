# -*- coding: utf-8 -*-
from keras.layers import Input, LSTM, Dense, concatenate, crf, RepeatVector, TimeDistributed, Multiply, Embedding
from keras.layers.core import Lambda
from keras.models import Model
from keras import backend as K
from gensim.models import Word2Vec
import numpy as np
from evaluation import raw_result_parser
from evaluation import tagging_util

class QAModel:
    def __init__(self):
        self.wordModel = Word2Vec.load('model_new_and_wiki')
        self.word_dim = 100
        self.question_len = 30
        self.evidence_len = 100
        self.model = None

    def word_vec(self, word):
        try:
            return self.wordModel[word]
        except KeyError:
            return np.zeros((self.word_dim,))

    def get_model(self):
        dropout_rate = 0.05
        D = 64
        # question layer
        question_input = Input(shape=(self.question_len, self.word_dim), name='q_input')
        # question_x = Embedding(output_dim=question_len,
        #               input_dim=word_dim,
        #               input_length=question_len)(question_input)
        q = LSTM(D, dropout=dropout_rate, return_sequences=True)(question_input)

        alpha_tanh = Dense(D, activation='tanh')(q)
        alpha = Dense(1, activation='softmax')(alpha_tanh)
        alpha_multiply = Multiply()([q, alpha])
        question_represent = Lambda(lambda x: K.sum(x, axis=1), name='question_representation')(alpha_multiply)
        # g1 layer
        qe_input = Input(shape=(self.evidence_len, 1), name='qe')
        # qe_layer = TimeDistributed(Embedding(input_dim=1, output_dim=1, name='qe_layer'))(qe_input)
        ee_input = Input(shape=(self.evidence_len, 1), name='ee')
        # ee_layer = TimeDistributed(Embedding(input_dim=1, output_dim=1, name='ee_layer'))(ee_input)
        # evidence layer
        evidence_input = Input(shape=(self.evidence_len, self.word_dim), name='e_input', dtype=np.float32)
        # what the fuck is g1 and g2 in the paper formula  10, don't care, ignore first
        question_represent_repeat = RepeatVector(self.evidence_len)(question_represent)
        evidence_x = concatenate([evidence_input, question_represent_repeat, qe_input, ee_input], name='input_e')
        # evidence_x = Embedding(output_dim=evidence_len,
        #                        input_dim=word_dim + 1,
        #                        input_length=evidence_len)(evidence)
        lstm_1 = LSTM(64, dropout=dropout_rate, return_sequences=True)(evidence_x)
        lstm_2 = LSTM(64, dropout=dropout_rate, return_sequences=True)(lstm_1)
        lstm_2_reverse = Lambda(lambda x: K.reverse(x, 0), name='lstm_reverse')(lstm_2)
        lstm_3 = LSTM(self.evidence_len, dropout=dropout_rate, return_sequences=True)(concatenate([lstm_1, lstm_2_reverse]))
        # CRF layer
        tp_layer = TimeDistributed(Dense(4))(lstm_3)
        crf_layer = crf.ChainCRF(name='crf_layer')
        crf_o = crf_layer(tp_layer)

        model = Model(inputs=[question_input, evidence_input, qe_input, ee_input], outputs=[crf_o])
        model.summary()
        model.compile(optimizer='rmsprop', loss=crf_layer.loss, loss_weights=[0.01])
        self.model = model
        return self.model

    def load_train_data(self, data_dir):
        data = raw_result_parser.load_train_data(data_dir)
        load_max = 100
        load_cur = 0
        for q_token, e_token, golden_label, g1, g2 in data:
            output_seq = np.zeros((self.evidence_len, 3))
            output_seq = np.hstack((output_seq, np.ones((self.evidence_len, 1))))
            input_q, input_e, input_qe, input_ee = self.get_input_q_e_g1_g2(q_token, e_token, g1, g2)
            for label_index in range(min(self.evidence_len, len(golden_label))):
                output_seq[label_index] = golden_label[label_index]
            yield input_q, input_e, input_qe, input_ee, output_seq
            load_cur += 1
            if load_cur > load_max:
                break

    ''' @:param the list of split question and evidence
        @:return a maxtric which col is the word_vec of the split word'''
    def get_input_q_e_g1_g2(self, question, evidence, g1, g2):
        input_q = np.zeros((self.question_len, self.word_dim))
        input_g1 = np.zeros((self.evidence_len, 1))
        input_e = np.ones((self.evidence_len, self.word_dim))
        input_g2 = np.ones((self.evidence_len, 1))
        for word_index in range(min(self.question_len, len(question))):
            word = question[word_index]
            input_q[word_index] = self.word_vec(word)
        for word_index in range(min(self.evidence_len, len(evidence))):
            word = evidence[word_index]
            input_e[word_index] = self.word_vec(word)
        for g1_index in range(min(self.evidence_len, len(g1))):
            input_e[g1_index] = g1[g1_index]
        for g2_index in range(min(self.evidence_len, len(g2))):
            input_e[g2_index] = g2[g2_index]
        return input_q, input_e, input_g1, input_g2

    def fit(self, input_qs, input_es, input_qe, input_ee, train_labels):
        self.model.fit([input_qs, input_es, input_qe, input_ee], [train_labels], epochs=10, batch_size=120)

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
        return self.model.predict([input_qs, input_es])

if __name__ == '__main__':
    qa = QAModel()
    qa.get_model()
    # for test_q, test_e, test_labels in qa.load_data('WebQA.v1.0/data/test.ir.json.gz'):
    #     print(test_q, test_e, test_labels)
    # test_q, test_e, test_labels = qa.load_data('WebQA.v1.0/data/test.ann.json.gz')
    iters = qa.load_train_data('WebQA.v1.0/data/training.json.gz')
    train_qs = []
    train_es = []
    train_qes = []
    train_ees = []
    train_labels = []
    for train_q, train_e, train_qe, train_ee, train_label in iters:
        train_qs.append(train_q)
        train_es.append(train_e)
        train_qes.append(train_qe)
        train_ees.append(train_ee)
        train_labels.append(train_label)

    train_qs = np.array(train_qs, dtype=np.float32)
    train_es = np.array(train_es, dtype=np.float32)
    train_qes = np.array(train_qes, dtype=np.float32)
    train_ees = np.array(train_ees, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)

    qa.fit(train_qs, train_es, train_qes, train_ees, train_labels)
    # qamodel.save_weights('qamodel.h5')
    # qa.load_model('qamodel.h5')
    # print(qa.model.evaluate([train_qs, train_es], [train_labels]))



