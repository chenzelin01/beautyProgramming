# -*- coding: utf-8 -*-
from keras.layers import Input, LSTM, Dense, concatenate, crf, RepeatVector, TimeDistributed, Multiply, Embedding
from keras.layers.core import Lambda
from keras.models import Model
from keras import backend as K
from gensim.models import Word2Vec
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8),
    device_count={'GPU': 1}
)
set_session(tf.Session(config=config))
import numpy as np
import sys
sys.path.append('../..')
from evaluation import raw_result_parser
from src.DBBASEParser import DBBASEParser


def find_index(list, word):
    try:
        return list.index(word)
    except:
        return -1

class PresentResult:
    def __init__(self, db_file, result_file, model):
        self.result_file = result_file
        self.qamodel = model
        self.db_parser = DBBASEParser(db_file)
    ''':param qs is the list of questions and es is the list of answers'''
    def get_labels(self, qs, es):
        input_qs = []
        input_es = []
        for q, e in zip(qs, es):
            input_q, input_e = self.qamodel.get_input_q_e(q, e)
            input_qs.append(input_q)
            input_es.append(input_e)
        return self.qamodel.predict(input_qs, input_es)

    def parse_db(self):
        iter = self.db_parser.iter_result()
        qs = []
        es = []
        ls = []
        max_iter = 1000
        cur = 0
        for q, e, l in iter:
            qs.append(q)
            es.append(e)
            ls.append(l)
            if cur > max_iter:
                break
            else:
                cur += 1
        pred_ls = self.get_labels(qs, es)
        return qs, es, ls, pred_ls

    def write_result(self):
        qs, es, labels, pred_labels = self.parse_db()
        correct = 0
        wrong = 0
        match = 0
        max_q_e = len(labels)
        with open(self.result_file, mode='w', encoding='utf-8') as rf:
            for q, e, l, pred in zip(qs, es, labels, pred_labels):
                pred = pred.tolist()
                b_index = find_index(pred, [1, 0, 0, 0])
                i_index = find_index(pred, [0, 1, 0, 0])
                pred = 0
                if b_index != -1 or i_index != -1:
                    for a in e:
                        tp = find_index(q, a)
                        if tp != -1:
                            pred = 1
                            break
                    # pred = 1
                if pred == l:
                    if pred == 1:
                        match += 1
                    correct += 1
                else:
                    wrong += 1
                line_words = [str(pred), str(l), "".join(q), "".join(e)]
                rf.writelines(" ".join(line_words))
        labels = np.array(labels)
        return match / np.sum(labels), correct / max_q_e, wrong / max_q_e

'''this model is the first native model implements https://arxiv.org/pdf/1607.06275.pdf'''
class QAModel:
    def __init__(self):
        self.wordModel = Word2Vec.load('../../model_new_and_wiki')
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
        with tf.device('/gpu:0'):
            dropout_rate = 0.05
            # question layer
            question_input = Input(shape=(self.question_len, self.word_dim), name='q_input')
            q = LSTM(self.question_len, dropout=dropout_rate, return_sequences=True)(question_input)

            alpha_tanh = Dense(self.question_len, activation='tanh')(q)
            alpha = Dense(1, activation='softmax')(alpha_tanh)
            alpha_multiply = Multiply()([q, alpha])
            question_represent = Lambda(lambda x: K.sum(x, axis=1), name='question_representation')(alpha_multiply)
            # evidence layer
            evidence_input = Input(shape=(self.evidence_len, self.word_dim), name='e_input', dtype=np.float32)
            # what the fuck is g1 and g2 in the paper formula  10, don't care, ignore first
            question_represent_repeat = RepeatVector(self.evidence_len)(question_represent)
            evidence_x = concatenate([evidence_input, question_represent_repeat], name='input_e')
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

            model = Model(inputs=[question_input, evidence_input], outputs=[crf_o])
            model.summary()
            model.compile(optimizer='rmsprop', loss=crf_layer.loss)
            self.model = model
            return self.model

    def load_train_data(self, data_dir):
        data = raw_result_parser.load_train_data(data_dir)
        # load_max = 100
        # load_cur = 0
        for q_token, e_token, golden_label, g1, g2 in data:
            output_seq = np.zeros((self.evidence_len, 3))
            output_seq = np.hstack((output_seq, np.ones((self.evidence_len, 1))))
            input_q, input_e = self.get_input_q_e(q_token, e_token)
            for label_index in range(min(self.evidence_len, len(golden_label))):
                output_seq[label_index] = golden_label[label_index]
            yield input_q, input_e, output_seq
            # load_cur += 1
            # if load_cur > load_max:
            #     break

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

    def fit(self, input_qs, input_es, train_labels):
        self.model.fit([input_qs, input_es], [train_labels], epochs=1, batch_size=120)

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

    def evaluation(self):
        p_hp = PresentResult(db_file='../../BoP2017_DBAQ_dev_train_data/BoP2017-DBQA.dev.txt',
                             result_file='../../BoP2017_DBAQ_dev_train_data/res.txt',
                             model=self)
        match, correct, wrong = p_hp.write_result()
        print(match)
        print(correct)
        print(wrong)

if __name__ == '__main__':
    qa = QAModel()
    # qa.get_model()
    # for test_q, test_e, test_labels in qa.load_data('WebQA.v1.0/data/test.ir.json.gz'):
    #     print(test_q, test_e, test_labels)
    # test_q, test_e, test_labels = qa.load_data('WebQA.v1.0/data/test.ann.json.gz')
    iters = qa.load_train_data('../../WebQA.v1.0/data/training.json.gz')
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

    qa.load_model('qamodel2_epoch4.h5')
    epochs = 14
    for i in range(5, epochs):
        qa.fit(train_qs, train_es, train_labels)
        model_name = 'qamodel2_epoch' + str(i)
        qa.save_model(model_name)
        qa.evaluation()
        print('epochs ', str(i), ' finished')
    # qa.load_model('qamodel.h5')
    # qamodel.save_weights('qamodel.h5')
    # qa.load_model('qamodel.h5')
    # print(qa.model.evaluate([train_qs, train_es], [train_labels]))



