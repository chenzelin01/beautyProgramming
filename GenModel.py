# -*- coding:utf-8 -*-
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import jieba
import logging
import os
import sys
import multiprocessing
from time import time
import re
if __name__ == '__main__':
    # pattern = re.compile("[\S]*(\s)*[\S]*")
    # this option is only can used in linux environment
    # jieba.enable_parallel(multiprocessing.cpu_count())
    t1 = time()
    training_txt_file = 'wiki_split'
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.getLogger(os.path.basename(sys.argv[0])).info("'running %s" % ' '.join(sys.argv))
    if os.path.isfile('model'):
        print('----- model have been existed so train the new txt based on the pre trained model -----')
        model = Word2Vec.load('model')
        model.train(LineSentence(training_txt_file), total_examples=model.corpus_count, epochs=model.iter)
        model.save('model')
    else:
        model = Word2Vec(LineSentence('../word2vec-master/resultbig.txt'), workers=multiprocessing.cpu_count())
        model.save('model')
    print("----- testing the training result -----")
    model = Word2Vec.load('model')
    sim = model.similarity("周杰伦", "林")
    print(sim)
    print("time using: ", time() - t1)
