# -*- coding:utf-8 -*-
import jieba
from QAModel2 import QAModel2
from DBBASEParser import DBBASEParser
import numpy as np


def find_index(list, word):
    try:
        return list.index(word)
    except:
        return -1

class PresentResult:
    def __init__(self, db_file, result_file, model_file):
        self.result_file = result_file
        self.qamodel = QAModel2()
        self.qamodel.load_model(model_file)
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
        max_iter = 100
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

if __name__=='__main__':
    p_result = PresentResult(db_file='BoP2017_DBAQ_dev_train_data/BoP2017-DBQA.train.txt',
                             result_file='BoP2017_DBAQ_dev_train_data/res.txt',
                             model_file='qamodel_0.2.h5')

    match, correct, wrong = p_result.write_result()
    print(match)
    print(correct)
    print(wrong)

    # q = "中西区圣安多尼学校是什么时候成立了校友会？"
    # q = list(jieba.cut(q))
    # ans = ["中西区圣安多尼学校（英文：Central And Western District Saint Anthony's School）于1963年创立，隶属鲍思高慈幼会中华会省。",
    #        "中西区圣安多尼学校是2005年9月17日成立校友会。",
    #     "2006年9月:完成所有收生程序，正式成为一所男女校。",
    #     "盾上方写着“中西区圣安多尼学校”的中英文字样，下方中央有一特式百合花，两旁有代表校名的英文简写\"CWSA\"，以及中文校训：“笃信博爱”。"]
    # qs = []
    # qes = []
    # for i in range(len(ans)):
    #     ans[i] = list(jieba.cut(ans[i]))
    # es = ans
    # # max_qe_ee_len = 100
    # # qe = np.zeros(max_qe_ee_len)
    # # for q_word_index in range(len(q)):
    # #     q_word = q[q_word_index]
    # #     for a in ans:
    # #         if find_index(a, q_word) is not -1:
    # #             qe[q_word_index] = 1
    # #             break
    # #
    # # ans_len = len(ans)
    # # ees = np.zeros((ans_len, max_qe_ee_len))
    # #
    # # index_i = 0
    # # while index_i < ans_len:
    # #     for ans_word_index in range(len(ans[index_i])):
    # #         ans_word = ans[index_i][ans_word_index]
    # #         for index_j in range(index_i+1, ans_len):
    # #             word_find_index = find_index(ans[index_j], ans_word)
    # #             if word_find_index is not -1:
    # #                 # the vector of finding answer
    # #                 ees[index_i][ans_word_index] = 1
    # #                 # the vector of where mactch answer
    # #                 ees[index_j][word_find_index] = 1
    # #     index_i += 1
    # #     qes.append(qe)
    #
    # qa = QAModel2()
    # input_qs = []
    # input_es = []
    # # input_qes = []
    # # input_ees = []
    # for input_e in es:
    #     input_q, input_e = qa.get_input_q_e(q, input_e)
    #     input_qs.append(input_q)
    #     input_es.append(input_e)
    #
    # input_qs = np.array(input_qs)
    # input_es = np.array(input_es)
    # qa.load_model('qamodel.h5')
    # result = qa.predict(input_qs, input_es)
    # print(result)

