# -*- coding:utf-8 -*-
''' this module is 2 change the WebQA json file into the format like pre-train '''
import json
import sys
sys.path.append('..')
from evaluation.ioutil import open_file
from evaluation.datapoint import DataPoint, Evidence, EecommFeatures
__all__ = ['iter_results']


def parse_line_label(line):
    data = json.loads(line)
    q_tokens = data[DataPoint.Q_TOKENS]
    evis = data[DataPoint.EVIDENCES]
    postive_evis = []
    negative_evis = []
    for evi in evis:
        if evi['type'] == 'positive':
            postive_evis.append(evi[Evidence.E_TOKENS])
        else:
            negative_evis.append(evi[Evidence.E_TOKENS])
    return q_tokens, postive_evis, negative_evis


def format_data(data_file):
    with open_file(data_file) as data_file:
        for line in data_file:
            q_tokens, postive_evis, negative_evis = parse_line_label(line)
            yield q_tokens, postive_evis, negative_evis


if __name__=='__main__':
    # max_iter = 300
    cur_line = 0
    lines = []
    with open('../BoP2017_DBAQ_dev_train_data/WebQAJson2Txt.txt', mode='w', encoding='utf-8') as f:
        for q_tokens, postive_evis, negative_evis in format_data('../WebQA.v1.0/data/training.json.gz'):
            q = "".join(q_tokens)
            for postive_evi in postive_evis:
                postive_evi = "".join(postive_evi)
                for negative_evi in negative_evis:
                    negative_evi = "".join(negative_evi)
                    line = "\t".join([q, postive_evi, negative_evi])
                    lines.append(line)
                    cur_line += 1
            # if cur_line > max_iter:
            #     break
        lines = '\n'.join(lines)
        f.writelines(lines)
    print(cur_line)
