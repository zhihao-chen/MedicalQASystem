from functools import reduce
import jieba
#from classfier import predict
from kbqa_module.kbqa import KBQA
from qa_pair_module.DatabaseOp import get_questions, get_answers
from qa_pair_module.similarityCal import get_similarity
from qa_pair_module.retrieval import bool_retrieval
from qa_pair_module.QAMatching import matching
import mysql.connector as mc


class Config:
    def __init__(self):
        self.mydb = mc.connect(host='localhost',
                               user='root',
                               password='123456789',
                               database='medical',
                               auth_plugin='mysql_native_password')
        self.mycursor = self.mydb.cursor()


def set_and_set(sets1, sets2):
    """
    求集合的交集
    :param sets1: Set
    :param sets2: Set
    :return: Set
    """
    return sets1.intersection(sets2)


def kbqa_search(entities, handler):
    """
    基于知识库的问答检索模块
    :param entities: 实体和意图
    :param handler: 类
    :return:
    """
    answer = handler.search(entities)
    return answer


def qa_pair_search(question, config):
    """
    问答对检索模块
    :param question: str
    :return: str
    """
    jieba.load_userdict('vocab.txt')
    #cls = predict(question)  # 预测科室
    cls = 'questions'
    # 初步检索候选答案
    ids = []
    words = jieba.cut(question)
    for word in words:
        idx_list = bool_retrieval(word, cls + "_index", config.mycursor)
        if idx_list:
            ids.append(idx_list)
    new_ids = list(reduce(set_and_set, ids))

    questions = get_questions(new_ids, cls, config.mycursor)  # 根据倒排表的结构得到候选问题
    candidate_q_ids, candidate_qs = get_similarity(question, new_ids, questions)  # 计算问题与候选答案的相似度，进一步缩小范围
    candidate_answers = get_answers(candidate_q_ids, config.mycursor)  # 根据候选答案id，从数据库中获取候选答案

    answer = ''
    if len(candidate_answers) == 1:
        if len(candidate_answers[0]) == 1:
            answer = candidate_answers[0][0][1]
    else:
        answer = matching(question, candidate_qs, candidate_answers)  # 通过QA matching模型得到最佳答案

    return answer


def recommend(query):
    """
    根据用户问题向用户返回答案
    :param query: str
    :return: str
    """
    config = Config()
    handler = KBQA()
    data = {}

    entity_intent = handler.entity_intent_reg(query)
    intent = entity_intent['intentions']
    answer = ''
    if intent != 'QA_matching':
        answer = kbqa_search(entity_intent, handler)
    elif intent == 'QA_matching' or answer == '':
        answer = qa_pair_search(query, config)
    text = "对不起，您的问题我暂时无法回答！我将记录您的问题，争取下次能给您答案！感谢您的使用！"

    data["question"] = query
    if answer:
        data["answer"] = answer
    else:
        with open('./DATA/user_questions/user_question.text', 'a', encoding='utf8') as f:
            f.write(query + '\n')
        data["answer"] = text
    return data
