"""this script is used to compute the similarity between question and answer using
the chapter 4 in http://aptnk.in/profile/papers/thesis-final-btech.pdf"""
from gensim.models import Word2Vec
import jieba
import numpy as np
import re

class SentenceSimilarity:
    def __init__(self):
        self.model = Word2Vec.load('model_new_and_wiki')
        self.signature_re = re.compile("[\s+\.\!\/_,$%^*(+\"\';:]+|[+——！，。？、~@#￥%……&*（）；：》《｛｝【】]+")

    def word_sim(self, word1, word2):
        try:
            return self.model.similarity(word1, word2)
        except KeyError:
            return 0

    def all_answer_score(self, question, answer_list):
        score_list = []
        for ans in answer_list:
            score_list.append(self.answer_score(question, ans))
        return score_list

    def answer_score(self, question, ans):
        alpha = 0.1
        miu = 0.5
        fi = 1.
        lambda_ = 0.25
        v = 0.125

        a_tokens = self.erase_signature(ans)
        q_tokens = self.erase_signature(question)
        question_map = []
        for q_word in q_tokens:
            q_word_edge = self.qword_match_answer(q_word, a_tokens)
            question_map.append(q_word_edge)
        question_map = np.array(question_map)
        try:
            E_total = np.mean(np.max(question_map, axis=1))
        except:
            ''' this except because the answer is only combine with signature so ignore it
                e.g 网舞四代的阪本奨悟身高多少？	）'''
            return -100
        S_total = np.mean(question_map)
        # compute K total
        theta_distance = np.argmax(question_map, axis=1)
        K_total = np.mean(np.sum(np.sign(np.diff(theta_distance))))
        n = len(a_tokens)
        m = len(q_tokens)
        delta_total = np.exp(-1 * alpha * (n - E_total * m)) - 1
        score = fi * E_total + lambda_ * S_total + miu * delta_total + v * K_total
        return score


    def qword_match_answer(self, qword, a_tokens):
        match_list = []
        for a_word in a_tokens:
            sim = self.word_sim(qword, a_word)
            match_list.append(sim)
        return match_list

    '''erase the signature in sentences and return the tokens after split'''
    def erase_signature(self, sentences):
        s_tokens = list(jieba.cut(sentences))
        sentence_len = len(s_tokens)
        cur_index = 0
        # erase the signature of the sentence
        while cur_index < sentence_len:
            word = s_tokens[cur_index]
            if self.signature_re.search(word) is not None:
                s_tokens.pop(cur_index)
                cur_index -= 1
                sentence_len -= 1
            cur_index += 1
        return s_tokens

    def evalution(self, max_iter=None):
        def parse_line(line):
            keys = line.split('\t')
            try:
                label = int(keys[0])
            except:
                label = int(keys[0][1])
            return keys[1], keys[2], label
        cur_iter = 0
        predict_sentence_buffer = []
        ls = []
        qs = []
        ans = []
        preds = []
        last_q = None
        with open("BoP2017_DBAQ_dev_train_data/BoP2017-DBQA.dev.txt", mode='r', encoding='utf-8') as f:
            line = f.readline()
            while line != -1 and len(line) > 0:
                q, a, l = parse_line(line)
                ls.append(l)
                qs.append(q)
                ans.append(a)
                if last_q is None:
                    last_q = q
                elif last_q != q:
                    predict_sentence_buffer.pop(0)
                    vec_list = self.all_answer_score(last_q, predict_sentence_buffer)
                    last_q = q
                    for v in vec_list:
                        preds.append(v)
                    predict_sentence_buffer = []
                predict_sentence_buffer.append(q)
                predict_sentence_buffer.append(a)
                line = f.readline()
                if max_iter is not None:
                    cur_iter += 1
                    if cur_iter > max_iter:
                        break

        with open("BoP2017_DBAQ_dev_train_data/Score.txt", mode='w', encoding='utf-8') as f:
            for pred, l, q, a in zip(preds, ls, qs, ans):
                f.writelines(str(pred))



if __name__=='__main__':
    # Q1 = "中西区圣安多尼学校是什么时候成立了校友会？"
    # A1 = "中西区圣安多尼学校（英文：Central And Western District Saint Anthony's School）于1963年创立，隶属鲍思高慈幼会中华会省。"
    # A2 = "中西区圣安多尼学校创建于1963年，为天主教鲍思高慈幼会属下小学，原是圣安多尼学校下午校，校舍座落于石塘咀薄扶林道69A。"
    # A3 = "2005年9月17日:成立校友会。"
    # A4 = "盾上方写着“中西区圣安多尼学校”的中英文字样，下方中央有一特式百合花，两旁有代表校名的英文简写\"CWSA\"，以及中文校训：“笃信博爱”。"
    # sim_hp = SentenceSimilarity()
    # print(sim_hp.all_answer_score(Q1, [A1, A2, A3, A4]))
    sim_hp = SentenceSimilarity()
    sim_hp.evalution()

