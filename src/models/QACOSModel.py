# -*- coding: utf-8 -*-
"""this module is implementation of matching question and answer using CNN to
   compute the cosine distance between question and answer citation of paper in
   https://arxiv.org/pdf/1508.01585.pdf"""

from keras.layers import Input, Conv1D, Dense, concatenate, crf, RepeatVector, GlobalAveragePooling1D, \
    TimeDistributed, Multiply, Embedding, Convolution1D, Bidirectional, GlobalMaxPool1D, Flatten, Dropout
from keras.layers.core import Lambda
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
import sys
sys.path.append('../..')
from src.Sentence2Matrix import Sentence2Matrix

import tensorflow as tf
from keras.backend import tensorflow_backend
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
    device_count={'GPU': 1}
)
config.gpu_options.allow_growth = True
tensorflow_backend.set_session(tf.Session(config=config))

class QACOSModel:
    def __init__(self):
        self.sentence_matrix_helper = Sentence2Matrix('../../model_new_and_wiki')
        self.margin = 0.2
        self.model = None
        self.word_dim = self.sentence_matrix_helper.word_dim
        self.sentence_len = self.sentence_matrix_helper.sentence_len
        self.r_dim = 120
        self.batch_m = 120
        self.db_file = "../../BoP2017_DBAQ_dev_train_data/BoP2017-DBQA.pre_train.txt"

    def compile_model(self):
        dropout_rate = 0.05
        input_layer = Input(shape=(self.sentence_len, self.word_dim), name='question')
        hidden_layer = Dropout(dropout_rate)(Dense(120, activation='tanh')(input_layer))
        cnn = Dropout(dropout_rate)(Conv1D(self.r_dim, kernel_size=3, activation='relu', name='cnn')(hidden_layer))
        pool = GlobalMaxPool1D(name='pooling_layer')(cnn)

        '''compute the cosine of r_q and r_a as the concatenate input of r_q and r_a'''
        def cosine_similarity(y_true, y_pred):
            y_true = K.l2_normalize(y_true, axis=1)
            y_pred = K.l2_normalize(y_pred, axis=1)
            return K.sum(y_true * y_pred, axis=1, keepdims=False)

        def cosine_ranking_loss(y_true, y_pred):
            q = y_pred[0::3]
            a_correct = y_pred[1::3]
            a_incorrect = y_pred[2::3]
            cosine_ = cosine_similarity(q, a_correct) - cosine_similarity(q, a_incorrect)
            maximum_ = K.maximum(K.zeros(shape=(self.batch_m,)), self.margin - cosine_)
            return K.mean(maximum_, axis=-1, keepdims=False)

        model = Model(inputs=[input_layer], outputs=[pool])
        model.summary()
        model.compile(optimizer='rmsprop', loss=cosine_ranking_loss)
        self.model = model
        return self.model

    ''':return each batch of question + correct answer + wrong answer'''
    def get_batch_generator(self):
        def parse_line(line):
            keys = line.split('\t')
            return keys[0], keys[1], keys[2]

        with open(self.db_file, mode='r', encoding='utf-8') as f:
            line = f.readline()
            input_sentences = []
            output_vectors = []
            batch_index = 0
            while line != -1 and len(line) > 0:
                question, correct, wrong = parse_line(line)
                q = self.sentence_matrix_helper.get_sentence_matrix(question)
                correct_answer = self.sentence_matrix_helper.get_sentence_matrix(correct)
                wrong_answer = self.sentence_matrix_helper.get_sentence_matrix(wrong)
                input_sentences.append(q)
                input_sentences.append(correct_answer)
                input_sentences.append(wrong_answer)
                for i in range(3):
                    output_vectors.append(np.zeros(self.r_dim, ))
                batch_index += 1
                if batch_index >= self.batch_m:
                    yield (np.array(input_sentences), np.array(output_vectors))
                    batch_index = 0
                    input_sentences = []
                    output_vectors = []
                line = f.readline()

    def fit(self, sample_num):
        self.model.fit_generator(self.get_batch_generator(), steps_per_epoch=int(sample_num / self.batch_m) - 1, epochs=1)

    def save_model(self, model_file):
        self.model.save_weights(model_file)

    def load_model(self, model_file):
        self.compile_model()
        self.model.load_weights(model_file)
        return self.model

    ''':param input_qes and input_ees should be type() as np.array'''
    def predict(self, input_sentences):
        sentences = []
        for s in input_sentences:
            s = self.sentence_matrix_helper.get_sentence_matrix(s)
            sentences.append(s)
        sentences = np.array(sentences)
        return self.model.predict([sentences])

    def evaluation(self, max_iter=None, write_into_file=True):
        def parse_line(line):
            keys = line.split('\t')
            try:
                label = int(keys[0])
            except:
                label = int(keys[0][1])
            return keys[1], keys[2], label

        def cosine_similarity(vec1, vec2):
            try:
                return np.dot(vec1.T, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            except:
                ''' this except because the answer is only combine with signature so ignore it
                    e.g 网舞四代的阪本奨悟身高多少？	）'''
                return -1

        def score_answer_vector(q_vec, a_vec_list):
            sim_score = []
            for a_v in a_vec_list:
                sim_score.append(cosine_similarity(q_vec, a_v))
            return sim_score

        def compare_score_and_label(score_list, label_list):
            rank_list = np.ones((len(score_list),), dtype=np.float32)
            label_list = np.array(label_list)
            score_list = np.array(score_list)
            sort_list = np.argsort(score_list)[::-1]
            r = 1
            for arg_index in sort_list:
                rank_list[arg_index] = r
                r += 1
            rank_list_cp = np.array(rank_list)
            rank_list_cp[label_list == 0] = 0
            s = np.sum(rank_list_cp)
            if s == 0:
                score = 0
            else:
                score = 1 / s
            return score, rank_list

        cur_iter = 0
        predict_sentence_buffer = []
        ls = []
        q_label_ls = []
        qs = []
        ans = []
        preds = []
        scores = []
        last_q = None
        q_num = 0
        total_scores = 0
        with open("../../BoP2017_DBAQ_dev_train_data/BoP2017-DBQA.dev.txt", mode='r', encoding='utf-8') as f:
            line = f.readline()
            while line != -1 and len(line) > 0:
                q, a, l = parse_line(line)
                ls.append(l)
                qs.append(q)
                ans.append(a)
                if last_q is None:
                    last_q = q
                    q_num += 1
                    predict_sentence_buffer.append(q)
                elif last_q != q:
                    last_q = q
                    q_num += 1
                    vec_list = self.predict(predict_sentence_buffer)
                    pred_score = score_answer_vector(vec_list[0], vec_list[1:len(vec_list)])
                    for s in pred_score:
                        scores.append(s)
                    score, rank = compare_score_and_label(pred_score, q_label_ls)
                    total_scores += score
                    for r in rank:
                        preds.append(r)
                    predict_sentence_buffer = []
                    q_label_ls = []
                    predict_sentence_buffer.append(q)
                q_label_ls.append(l)
                predict_sentence_buffer.append(a)
                line = f.readline()
                if max_iter is not None:
                    cur_iter += 1
                    if cur_iter > max_iter:
                        break
            # predict the last question
            vec_list = self.predict(predict_sentence_buffer)
            pred_score = score_answer_vector(vec_list[0], vec_list[1:len(vec_list)])
            for s in pred_score:
                scores.append(s)
            score, rank = compare_score_and_label(pred_score, q_label_ls)
            total_scores += score
            for r in rank:
                preds.append(r)

        ret = total_scores / q_num
        print(ret)
        if write_into_file:
            print('writing to file')
            detail_lines = []
            scores_lines = []
            for pred, l, q, a, score in zip(preds, ls, qs, ans, scores):
                line = "\t".join([str(pred), str(l), q, a])
                detail_lines.append(line)
                scores_lines.append(str(score))
            with open("../../BoP2017_DBAQ_dev_train_data/QACOSResult_model4.txt", mode='w',
                      encoding='utf-8') as f:
                # print(scores_lines)
                f.writelines("\n".join(scores_lines))
            with open("../../BoP2017_DBAQ_dev_train_data/QACOSResult.txt", mode='w', encoding='utf-8') as f:
                # print(detail_lines)
                f.writelines(detail_lines)
        return ret

if __name__=='__main__':
    qacos = QACOSModel()
    qacos.compile_model()
    epochs = 20
    for i in range(epochs):
        # 224160
        # 798198
        qacos.fit(224160)
        print('----- QACOSModel maxpool margin .009 sgd epoch', str(i), '-----')
        score = qacos.evaluation(5000, write_into_file=False)
        model_name = 'QACOSModel_margin.009_sgd_epoch' + str(i) +'_score_' + str(score)
        qacos.save_model(model_name)
    # qacos.load_model('QACOSModel_epoch8_score_0.706349206349')
    # qacos.evaluation()