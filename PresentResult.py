# -*- coding:utf-8 -*-
import jieba
from QAModel import QAModel
from DBModel import DBModel
if __name__=='__main__':
    q = "我很好奇dolls特刑部队的队长是谁？"
    ans = ["法务省 特别死刑执行刑务官部 第一部队 总队长: 御子柴笑太,副部长:式部清寿, 队员：藤堂羽沙希",
         "由负责故事设计的咲木音穂（saki otoo，12月6日）以及负责绘画的中村友美（nakamura tomomi，3月17日）两人组成）连载。",
         "这个人身上充满了复杂的矛盾感，却不会显得不协调。"]

    # q = list(jieba.cut(q))
    # e = list(jieba.cut(e))
    db_model = DBModel()
    db_model.load_model('db_model.h5')
    qs = []
    es = []
    for e in ans:
        qs.append(q)
    result = db_model.predict_by_sentences(qs, ans)
    print(result)
