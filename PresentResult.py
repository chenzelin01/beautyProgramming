# -*- coding:utf-8 -*-
import jieba
from QAModel import QAModel
from DBModel import DBModel
import numpy as np

def find_index(list, word):
     try:
        return list.index(word)
     except:
         return -1


if __name__=='__main__':
    q = "中西区圣安多尼学校是什么时候成立了校友会？"
    q = list(jieba.cut(q))
    ans = ["中西区圣安多尼学校（英文：Central And Western District Saint Anthony's School）于1963年创立，隶属鲍思高慈幼会中华会省。",
           "中西区圣安多尼学校是2005年9月17日成立校友会。",
        "2006年9月:完成所有收生程序，正式成为一所男女校。",
        "盾上方写着“中西区圣安多尼学校”的中英文字样，下方中央有一特式百合花，两旁有代表校名的英文简写\"CWSA\"，以及中文校训：“笃信博爱”。"]
    qs = []
    qes = []
    for i in range(len(ans)):
        ans[i] = list(jieba.cut(ans[i]))
    es = ans
    max_qe_ee_len = 100
    qe = np.zeros(max_qe_ee_len)
    for q_word_index in range(len(q)):
        q_word = q[q_word_index]
        for a in ans:
            if find_index(a, q_word) is not -1:
                qe[q_word_index] = 1
                break

    ans_len = len(ans)
    ees = np.zeros((ans_len, max_qe_ee_len))

    index_i = 0
    while index_i < ans_len:
        for ans_word_index in range(len(ans[index_i])):
            ans_word = ans[index_i][ans_word_index]
            for index_j in range(index_i+1, ans_len):
                word_find_index = find_index(ans[index_j], ans_word)
                if word_find_index is not -1:
                    # the vector of finding answer
                    ees[index_i][ans_word_index] = 1
                    # the vector of where mactch answer
                    ees[index_j][word_find_index] = 1
        index_i += 1
        qes.append(qe)

    qa = QAModel()
    input_qs = []
    input_es = []
    input_qes = []
    input_ees = []
    for input_e, input_qe, input_ee in zip(es, qes, ees):
        input_q, input_e, input_qe, input_ee = qa.get_input_q_e_g1_g2(q, input_e, input_qe, input_ee)
        input_qs.append(input_q)
        input_es.append(input_e)
        input_qes.append(input_qe)
        input_ees.append(input_ee)

    input_qs = np.array(input_qs)
    input_es = np.array(input_es)
    input_qes = np.array(input_qes)
    input_ees = np.array(input_ees)
    qa.load_model('qamodel_0.1.h5')
    result = qa.predict(input_qs, input_es, input_qes, input_ees)
    print(result)

