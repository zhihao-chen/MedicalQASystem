import numpy as np
import re
import os
from gensim.models import Doc2Vec, Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

cur_dir = 'E:/PycharmProject/MedicalQASystem/app/qa_pair_module/'


def simlarityCalu(vector1, vector2):
    """
    计算两个句子的相似度得分
    :param vector1: Vector
    :param vector2: Vector
    :return:
    """
    vector1Mod = np.sqrt(vector1.dot(vector1))
    vector2Mod = np.sqrt(vector2.dot(vector2))
    if vector2Mod != 0 and vector1Mod != 0:
        simlarity = (vector1.dot(vector2)) / (vector1Mod * vector2Mod)
    else:
        simlarity = 0
    return simlarity


def doc2vec(text, model):
    """
    得到句子向量
    :param text:str
    :param model:
    :return: vector
    """
    import jieba

    doc = list(jieba.cut(text))

    start_alpha = 0.01
    infer_epoch = 1000

    doc_vec_all = model.infer_vector(doc, alpha=start_alpha, steps=infer_epoch)
    return doc_vec_all


def get_char_pos(string, char):
    chPos = []
    try:
        chPos = list(((pos) for pos, val in enumerate(string) if(val == char)))
    except:
        pass
    return chPos


def word2vec(text, model):
    """
    word2vec向量表示
    :param text:
    :param model:
    :return:
    """
    from jieba import analyse

    wordvec_size = 192
    word_vec_all = np.zeros(wordvec_size)

    data = ' '.join(analyse.extract_tags(text))
    space_pos = get_char_pos(data, ' ')
    first_word = data[0:space_pos[0]]
    if model.__contains__(first_word):
        word_vec_all = word_vec_all + model[first_word]

    for i in range(len(space_pos) - 1):
        word = data[space_pos[i]:space_pos[i + 1]]
        if model.__contains__(word):
            word_vec_all = word_vec_all + model[word]
    return word_vec_all


class ResultInfo(object):
    """
    存储检索信息
    """
    def __init__(self, que_id, score, text):
        self.que_id = que_id
        self.score = score
        self.text = text


def get_similarity(query, que_ids, questions):
    """
    相似度检索
    :param query: str
    :param que_ids: List(int)
    :param questions: List(str)
    :return: List(int)
    """
    model_path_1 = os.path.join(cur_dir, 'DATA/zhiwiki_news.doc2vec')
    model_path_2 = os.path.join(cur_dir, 'DATA/zhiwiki_news.word2vec')

    model_1 = Doc2Vec.load(model_path_1)
    model_2 = Word2Vec.load(model_path_2)

    text = re.sub("[，。；：？“”‘’【】、！]+", "", query)

    temp = []
    index = 0
    for sentence in questions:
        # 得到doc2vec向量
        a1 = doc2vec(text, model_1)
        a2 = doc2vec(sentence, model_1)
        # 得到word2vec向量
        b1 = word2vec(text, model_2)
        b2 = word2vec(sentence, model_2)
        # 合并两个向量，求其平均值，作为新的表示向量
        p1 = np.divide(np.add(a1, b1), 2)
        p2 = np.divide(np.add(a2, b2), 2)
        # score = simlarityCalu(p1, p2)
        score = cosine_similarity([p1], [p2])[0]
        if score >= 0.7:
            temp.append(ResultInfo(que_ids[index], score, questions[index]))
        index += 1
    temp.sort(key=lambda k: k.score, reverse=True)

    k = 0
    result_ids = []
    result_texts = []
    for res in temp:
        if k < 5:
            result_ids.append(res.que_id)
            result_texts.append(res.text)
        else:
            break
        k += 1
    return result_ids, result_texts
