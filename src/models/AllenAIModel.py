# -*- coding: utf-8 -*-
from keras.layers import Input, LSTM, Dense, concatenate, crf, RepeatVector,\
    TimeDistributed, Multiply, Embedding, Convolution1D, Bidirectional, MaxPooling1D, Flatten
from keras.layers.core import Lambda
from keras.models import Model
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from gensim.models import Word2Vec
import numpy as np
from evaluation import raw_result_parser
from evaluation import tagging_util
from src.Sentence2Matrix import Sentence2Matrix


'''this model is a model implemented https://arxiv.org/pdf/1511.04108.pdf'''
class AllenAIModel:
    def __init__(self):
        self.sentence_matrix_helper = Sentence2Matrix("../../model_new_and_wiki")
        self.sentence_len = self.sentence_matrix_helper.sentence_len
        self.word_dim = self.sentence_matrix_helper.word_dim
        self.model = None
        self.margin = 0.2
        self.db_file = "../../BoP2017_DBAQ_dev_train_data/BoP2017-DBQA.pre_train.txt"

    def compile_model(self):
        input_layer = Input(shape=(self.sentence_len, self.word_dim), name="input_layer")
        LSTM_layer = LSTM(128, return_sequences=True)(input_layer)
        LSTM_layer_reserve = Lambda(lambda x: K.reverse(x, axes=-1))(LSTM_layer)
        LSTM_layer = LSTM(128, return_sequences=True, name='biLSTM')(LSTM_layer_reserve)

        # add softmax weight
        alpha_tanh = Dense(self.sentence_len, activation='tanh')(LSTM_layer)
        alpha = Dense(1, activation='softmax')(alpha_tanh)
        softmax_weight = Multiply()([LSTM_layer, alpha])

        conv_layer = Convolution1D(filters=128, kernel_size=3, name='conv_layer')(softmax_weight)
        pooling_layer = Flatten()(MaxPooling1D(pool_size=4)(conv_layer))
        hidden_layer = Dense(128)(pooling_layer)
        out_layer = Dense(128, activation='softmax')(hidden_layer)
        self.model = Model(inputs=[input_layer], outputs=[out_layer])

        def cosine_similarity(y_true, y_pred):
            y_true = K.l2_normalize(y_true, axis=1)
            y_pred = K.l2_normalize(y_pred, axis=1)
            return K.sum(y_true * y_pred, axis=1, keepdims=False)

        def cosine_ranking_loss(y_true, y_pred):
            q = y_pred[0::3]
            a_correct = y_pred[1::3]
            a_incorrect = y_pred[2::3]
            return K.mean(
                K.maximum(0., self.margin - cosine_similarity(q, a_correct) + cosine_similarity(q, a_incorrect)) - y_true[
                    0] * 0, axis=-1)

        self.model.compile(optimizer='rmsprop', loss=cosine_ranking_loss)
        self.model.summary()

    ''':return each batch of question + correct answer + wrong answer'''
    def get_batch_generator(self, max_iter=None):
        def parse_line(line):
            keys = line.split('\t')
            return keys[0], keys[1], keys[2]
        with open(self.db_file, mode='r', encoding='utf-8') as f:
            line = f.readline()
            # cur_iter = 0
            while line != -1 and len(line) > 0:
                question, correct, wrong = parse_line(line)
                q = self.sentence_matrix_helper.get_sentence_matrix(question)
                correct_answer = self.sentence_matrix_helper.get_sentence_matrix(correct)
                wrong_answer = self.sentence_matrix_helper.get_sentence_matrix(wrong)
                yield (np.array([q, correct_answer, wrong_answer]),
                       np.array([np.zeros(128,), np.zeros(128,), np.zeros(128,)]))
                line = f.readline()
                # if max_iter is not None:
                #     cur_iter += 1
                #     if cur_iter > max_iter:
                #         break

    def fit(self, max_iters):
        self.model.fit_generator(self.get_batch_generator(max_iters), steps_per_epoch=max_iters, epochs=1)

    def save_model(self, model_file):
        self.model.save_weights(model_file)

    def load_model(self, model_file):
        try:
            self.compile_model()
            self.model.load_weights(model_file)
            return self.model
        except:
            print("warning the model_file is not the model belong this module")
            return None
    ''':param input_qes and input_ees should be type() as np.array'''
    def predict(self, input_sentences):
        sentences = []
        for s in input_sentences:
            s = self.sentence_matrix_helper.get_sentence_matrix(s)
            sentences.append(s)
        return self.model.predict([sentences])

    # def envaluate(self, max_iter=None):
    #     def parse_line(line):
    #         keys = line.split('\t')
    #         try:
    #             label = int(keys[0])
    #         except:
    #             label = int(keys[0][1])
    #         return keys[0], keys[1], label
    #     cur_iter = 0
    #     with open("../../BoP2017_DBAQ_dev_train_data/BoP2017-DBQA.pre_train.txt", mode='r', encoding='utf-8') as f:
    #         line = f.readline()
    #         while line != -1 and len(line) > 0:
    #             q, a, l = parse_line(line)
    #
    #         def cosine_similarity(y_true, y_pred):
    #             y_true = K.l2_normalize(y_true, axis=1)
    #             y_pred = K.l2_normalize(y_pred, axis=1)
    #             return K.sum(y_true * y_pred, axis=1, keepdims=False)

if __name__ == '__main__':
    m = AllenAIModel()
    m.compile_model()
    m.fit(225390)
    m.save_model("AllenAIModel.h5")



