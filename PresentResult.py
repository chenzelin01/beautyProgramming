# -*- coding:utf-8 -*-
import jieba
from QAModel import QAModel
from DBModel import DBModel
if __name__=='__main__':
    q = "南红玛瑙主要产地是哪？"
    e = "南红很早前就被人们开采利用了，玛瑙的古称“赤玉”就应包括南红玛瑙。"
    q = list(jieba.cut(q))
    e = list(jieba.cut(e))
    qamodel = QAModel()
    qamodel.load_model('qamodel.h5')
    qs = []
    es = []
    iq, ie = qamodel.get_input_q_and_e(q, e)
    qs.append(iq)
    es.append(ie)
    result = qamodel.predict(qs, es)
    print(q)
    print(e)
    print(result)
