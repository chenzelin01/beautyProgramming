"""this script is used to compute the similarity of sentence"""
from gensim.models import Word2Vec
import jieba
import numpy as np
class SentenceSimilarity:
    def __init__(self):
        self.model = Word2Vec.load('model')
        self.signature_list = ['.', '。', ',', '，', '?', '？']
    def word_sim(self, word1, word2):
        try:
            return self.model.similarity(word1, word2)
        except KeyError:
            return 0
    '''@:param sentence similarity'''
    def sentence_sim(self,sentence_1, sentence_2):
        s1 = jieba.cut(sentence_1)
        s1 = list(s1)
        s2 = jieba.cut(sentence_2)
        s2 = list(s2)
        word_set = set(s1)
        word_set.union(s2)
        # word_set.remove(self.signature_list)
        s1_vec = []
        s2_vec = []
        for word in word_set:
            score1 = 0
            score2 = 0
            for word1 in s1:
                tp = self.word_sim(word, word1)
                score1 = max(score1, tp)
            for word2 in s2:
                tp = self.word_sim(word, word2)
                score2 = max(score2, tp)
            s1_vec.append(score1)
            s2_vec.append(score2)
        s1_vec = np.array(s1_vec)
        s2_vec = np.array(s2_vec)
        return np.dot(s1_vec.T, s2_vec) / np.linalg.norm(s1_vec) * np.linalg.norm(s2_vec)
if __name__=='__main__':
    Q1 = "中西区圣安多尼学校是什么时候成立了校友会？"
    A1 = "中西区圣安多尼学校（英文：Central And Western District Saint Anthony's School）于1963年创立，隶属鲍思高慈幼会中华会省。"
    A2 = "中西区圣安多尼学校创建于1963年，为天主教鲍思高慈幼会属下小学，原是圣安多尼学校下午校，校舍座落于石塘咀薄扶林道69A。"
    A3 = "2005年9月17日:成立校友会。"
    A4 = "盾上方写着“中西区圣安多尼学校”的中英文字样，下方中央有一特式百合花，两旁有代表校名的英文简写\"CWSA\"，以及中文校训：“笃信博爱”。"
    sim_hp = SentenceSimilarity()
    print(sim_hp.sentence_sim(Q1, A1))
    print(sim_hp.sentence_sim(Q1, A2))
    print(sim_hp.sentence_sim(Q1, A3))
    print(sim_hp.sentence_sim(Q1, A4))
