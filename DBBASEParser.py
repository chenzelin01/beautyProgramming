# -*- coding:utf-8 -*-
import jieba
import re

class DBBASEParser:
    def __init__(self, db_file):
        self.db_file = db_file

    def parse_line(self, line):
        keys = line.split('\t')
        try:
            label = int(keys[0])
        except:
            label = int(keys[0][1])
        q = jieba.cut(keys[1])
        e = jieba.cut(keys[2])
        return list(q), list(e), label

    ''':return question answer and label'''
    def iter_result(self):
        with open(self.db_file, encoding='utf-8') as db_f:
            line = db_f.readline()
            while line != -1 and len(line) > 0:
                question, evidence, label = self.parse_line(line)
                yield question, evidence, label
                line = db_f.readline()

if __name__ == '__main__':
    db_file = 'BoP2017_DBAQ_dev_train_data/BoP2017-DBQA.train.txt'
    db_parser = DBBASEParser(db_file)
    for q, e, l in db_parser.iter_result():
        print(q, e, l)
