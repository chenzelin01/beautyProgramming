from gensim.models import Word2Vec
import jieba
import numpy as np
import re


'''the class is a helper to transfer sentences into vector and erase the signature in
the sentences'''
class Sentence2Matrix:
    def __init__(self, word2vec_file):
        self.wordModel = Word2Vec.load(word2vec_file)
        self.word_dim = 100
        self.sentence_len = 64
        self.model = None
        self.signature_re = re.compile("[\s+\.\!\/_,$%^*(+\"\';:]+|[+——！，。？、~@#￥%……&*（）；：》《｛｝【】]+")

    def word_vec(self, word):
        try:
            return self.wordModel[word]
        except KeyError:
            return np.zeros((self.word_dim,))

    def get_sentence_matrix(self, sentence):
        if type(sentence) is not list:
            sentence_split = list(jieba.cut(sentence))
        else:
            sentence_split = sentence
        sentence_len = len(sentence_split)
        sentence_matrix = np.zeros((self.sentence_len, self.word_dim))
        cur_index = 0
        # erase the signature of the sentence
        while cur_index < sentence_len:
            word = sentence_split[cur_index]
            if self.signature_re.search(word) is not None:
                sentence_split.pop(cur_index)
                cur_index -= 1
                sentence_len -= 1
            cur_index += 1
        for word_index in range(min(len(sentence_split), self.sentence_len)):
            word_vec = self.word_vec(sentence_split[word_index])
            sentence_matrix[word_index] = word_vec
        return sentence_matrix

if __name__ == "__main__":
    test_str = "香港会议展览中心（简称会展；英语：Hong Kong Convention and Exhibition Centre，缩写：HKCEC）是香港的主要大型会议及展览场地，位于香港岛湾仔北岸，是香港地标之一；由香港政府及香港贸易发展局共同拥有，由新创建集团的全资附属机构香港会议展览中心（管理）有限公司管理。"
    mSentence2Matrix = Sentence2Matrix(word2vec_file="../model_new_and_wiki")
    print(mSentence2Matrix.get_sentence_matrix(test_str).shape)