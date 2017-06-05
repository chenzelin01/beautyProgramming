# -*- coding: utf-8 -*-
from keras.layers import Input, LSTM, Dense, concatenate, crf, RepeatVector,\
    TimeDistributed, Multiply, Embedding, Convolution1D, Bidirectional, GlobalAveragePooling1D, Flatten, Dropout
from keras.layers.core import Lambda
from keras.models import Model
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


''' this model is a model implemented https://arxiv.org/pdf/1511.04108.pdf
    with LSTM/CNN with attention to the question and shared the cnn layer'''
class AllenAIModel:
    def __init__(self):
        self.sentence_matrix_helper = Sentence2Matrix("../../model_new_and_wiki")
        self.sentence_len = self.sentence_matrix_helper.sentence_len
        self.word_dim = self.sentence_matrix_helper.word_dim
        self.model = None
        self.margin = 0.2
        self.db_file = "../../BoP2017_DBAQ_dev_train_data/BoP2017-DBQA.pre_train.txt"
        self.batch_m = 120
        self.output_dim = 128

    def compile_model(self):
        def cosine_similarity(x):
            q_vec, a_vec = x
            q_vec = K.l2_normalize(q_vec, axis=1)
            a_vec = K.l2_normalize(a_vec, axis=1)
            return K.sum(q_vec * a_vec, axis=1, keepdims=True)

        def cosine_ranking_loss(y_true, y_pred):
            cos_true = y_pred[0::2]
            cos_true = K.reshape(cos_true, shape=(-1,))
            cos_false = y_pred[1::2]
            cos_false = K.reshape(cos_false, shape=(-1,))
            cosine_ = cos_true - cos_false
            maximum_ = K.maximum(K.zeros(shape=(self.batch_m,)), self.margin - cosine_)
            return K.mean(maximum_, axis=-1, keepdims=False)

        dropout_rate = 0.05
        q_input = Input(shape=(self.sentence_len, self.word_dim), name="question_input_layer")
        question_answer_shared_biLSTM = Bidirectional(
            LSTM(self.output_dim, dropout=dropout_rate, return_sequences=True)
        )
        # question represent
        q_lstm = question_answer_shared_biLSTM(q_input)
        share_conv_layer = Convolution1D(
                filters=self.output_dim,
                kernel_size=3, name='share_conv_layer',
                activation='tanh')
        q_pooling_layer = GlobalAveragePooling1D(
            name='question_pooling_layer')(Dropout(dropout_rate)(share_conv_layer(q_lstm)))
        q_output_layer = Lambda(lambda x: K.reshape(x, (-1, self.output_dim,)), name='q_output_layer')(q_pooling_layer)

        # answer representation in lstm layer
        a_input = Input(shape=(self.sentence_len, self.word_dim), name="answer_input_layer")
        a_lstm = question_answer_shared_biLSTM(a_input)

        # the attention of question using the out put of q_lstm
        # add softmax weight
        alpha_tanh = Dense(self.sentence_len, activation='tanh')(concatenate([q_lstm, a_lstm]))
        alpha = Dense(1, activation='softmax')(alpha_tanh)
        softmax_weight = Multiply()([a_lstm, alpha])


        a_pooling_layer = GlobalAveragePooling1D(
            name='answer_pooling_layer'
        )(Dropout(dropout_rate)(share_conv_layer(softmax_weight)))
        # )(a_conv_layer)
        a_output_layer = Lambda(lambda x: K.reshape(x, (-1, self.output_dim,)), name='a_output_layer')(a_pooling_layer)

        cosine_layer = Lambda(function=cosine_similarity, name='cosine_layer')([q_output_layer, a_output_layer])
        # hidden_layer = Dropout(dropout_rate)(Dense(128, activation='tanh')(pooling_layer))
        # out_layer = Dropout(dropout_rate)(Dense(128, activation='tanh')(hidden_layer))
        self.model = Model(inputs=[q_input, a_input], outputs=[cosine_layer])
        self.model.compile(optimizer='rmsprop', loss=cosine_ranking_loss)

        # self.model.summary()

    ''':return each batch of question + correct answer + wrong answer'''
    def get_batch_generator(self):
        def parse_line(line):
            keys = line.split('\t')
            return keys[0], keys[1], keys[2]
        with open(self.db_file, mode='r', encoding='utf-8') as f:
            line = f.readline()
            input_qs = []
            input_as = []
            output_vectors = []
            batch_index = 0
            while line != -1 and len(line) > 0:
                question, correct, wrong = parse_line(line)
                q = self.sentence_matrix_helper.get_sentence_matrix(question)
                correct_answer = self.sentence_matrix_helper.get_sentence_matrix(correct)
                wrong_answer = self.sentence_matrix_helper.get_sentence_matrix(wrong)
                input_qs.append(q)
                input_qs.append(q)
                input_as.append(correct_answer)
                input_as.append(wrong_answer)
                for i in range(2):
                    output_vectors.append(np.zeros(1,))
                batch_index += 1
                if batch_index >= self.batch_m:
                    yield ([np.array(input_qs), np.array(input_as)], [np.array(output_vectors)])
                    batch_index = 0
                    input_qs = []
                    input_as = []
                    output_vectors = []
                line = f.readline()

    def fit(self, sample_num):
        self.model.fit_generator(self.get_batch_generator(), steps_per_epoch=sample_num / self.batch_m, epochs=1)

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
    def predict(self, questions, answers):
        input_qs = []
        input_as = []
        for q, a in zip(questions, answers):
            q = self.sentence_matrix_helper.get_sentence_matrix(q)
            a = self.sentence_matrix_helper.get_sentence_matrix(a)
            input_qs.append(q)
            input_as.append(a)
        input_qs = np.array(input_qs)
        input_as = np.array(input_as)
        return self.model.predict([input_qs, input_as])

    def evaluation(self, max_iter=None, write_into_file=True):
        def parse_line(line):
            keys = line.split('\t')
            try:
                label = int(keys[0])
            except:
                label = int(keys[0][1])
            return keys[1], keys[2], label

        def compare_score_and_label(score_list, label_list):
            rank_list = np.ones((len(score_list),), dtype=np.int32)
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
        q_buffer = []
        a_buffer = []
        ls = []
        q_label_ls = []
        qs = []
        ans = []
        preds = []
        scores = []
        last_q = None
        total_scores = 0
        q_num = 0
        with open("../../BoP2017_DBAQ_dev_train_data/BoP2017-DBQA.dev.txt", mode='r', encoding='utf-8') as f:
            line = f.readline()
            while line != -1 and len(line) > 0:
                q, a, l = parse_line(line)
                ls.append(l)
                qs.append(q)
                ans.append(a)
                # predict once when question is changed
                if last_q is None:
                    last_q = q
                    q_num += 1
                elif last_q != q:
                    last_q = q
                    q_num += 1
                    score_list = self.predict(q_buffer, a_buffer)
                    score_list = [s[0] for s in score_list]
                    for s in score_list:
                        scores.append(s)
                    score, rank = compare_score_and_label(score_list, q_label_ls)
                    total_scores += score
                    for r in rank:
                        preds.append(r)
                    q_buffer = []
                    a_buffer = []
                    q_label_ls = []

                q_label_ls.append(l)
                q_buffer.append(q)
                a_buffer.append(a)
                line = f.readline()
                if max_iter is not None:
                    cur_iter += 1
                    if cur_iter > max_iter:
                        break

            # predict the last question
            score_list = self.predict(q_buffer, a_buffer)
            score_list = [s[0] for s in score_list]
            for s in score_list:
                scores.append(s)
            q_correct, rank = compare_score_and_label(score_list, q_label_ls)
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
            with open("../../BoP2017_DBAQ_dev_train_data/allenAI5ScoreResult.txt", mode='w', encoding='utf-8') as f:
                # print(scores_lines)
                f.writelines("\n".join(scores_lines))
            with open("../../BoP2017_DBAQ_dev_train_data/allenAI5Result.txt", mode='w', encoding='utf-8') as f:
                # print(detail_lines)
                f.writelines(detail_lines)
        return ret


if __name__ == '__main__':
    m = AllenAIModel()
    m.compile_model()
    epochs = 20
    for epoch in range(epochs):
        # 224160
        m.fit(224160)
        score = m.evaluation(5000, write_into_file=False)
        model_save_name = "AllenAI5_aver_attention_epoch" + str(epoch) + '_' + str(score)
        m.save_model(model_save_name)
        print('AllenAI5 aver_attention epoch', str(epoch), 'finished')



