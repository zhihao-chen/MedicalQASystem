#问答匹配模块

import os
import numpy as np
import tensorflow as tf
from qa_pair_module.cnn_attention_model import CNNAttention

cur_dir = 'E:/PycharmProject/MedicalQASystem/app/qa_pair_module/'
word2vec_model_path = os.path.join(cur_dir, 'model/qa_char_1.txt')
save_path = os.path.join(cur_dir, 'model/best_validation')


class Config:
    def __init__(self):
        self.EMBEDDING_DIM = 200  # 词向量维度
        self.MAX_SEQ_LENGTH = 400  # 最大序列长度
        self.KERNEL_SIZE = [2, 3, 4]  # 卷积核大小
        self.NUM_FILTER = 800  # 卷积核数目
        self.DROPOUT_KEEP_PROB = 0.5  # dropout保留比例
        self.LEARNING_RATE = 1e-3  # 学习率
        self.MARGIN_VALUE = 0.05  # margin value
        self.L2_REG = 0.1


def load_embedding(path):
    vocab = []
    embed = []
    fp = open(path, 'r', encoding='utf8')
    header = fp.readline().strip()
    # print(header)
    vocab_size = int(header.split(' ')[0])
    word_dim = int(header.split(' ')[1])
    vocab.append("unk")
    embed.append([0] * word_dim)
    for new_line in fp:
        row = new_line.strip().split(' ')
        vocab.append(row[0])
        emb = [float(r) for r in row[1:]]
        embed.append(emb)
    # print('loaded word2vec')
    # print('The shape of embedding: ', (vocab_size, word_dim))
    fp.close()
    return vocab, embed, vocab_size, word_dim


def word_to_id(content, vocab):
    char_to_index = dict((c, i) for i, c in enumerate(vocab))
    data_id = []
    temp = [0] * 400
    a = 0
    for x in content:
        if x in char_to_index:
            temp[a] = char_to_index[x]
        else:
            temp[a] = char_to_index['unk']
        if a >= 400 - 1:
            break
        a += 1
    data_id.append(temp)

    return np.array(data_id)


def matching(str_q, candidate_q, candidate_a):
    """
    通过QA matching 模型从候选答案中选出top 1答案作为最终答案返回
    :param str_q: str
    :param candidate_q: List(str)
    :param candidate_a: List(List(tuple(int,str)))
    :return: str
    """
    config = Config()
    vocab, embedded, vocab_size, embed_dim = load_embedding(word2vec_model_path)
    model = CNNAttention(config.MAX_SEQ_LENGTH,
                         np.array(embedded),
                         config.EMBEDDING_DIM,
                         config.KERNEL_SIZE,
                         config.NUM_FILTER,
                         config.DROPOUT_KEEP_PROB,
                         config.MARGIN_VALUE,
                         config.LEARNING_RATE,
                         config.L2_REG)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, save_path)

        input_q = word_to_id(str_q, vocab)

        scores = []
        for i in range(len(candidate_a)):

            vec_q = word_to_id(candidate_q[i], vocab)
            temp = []
            j = 0
            for idx, ans in candidate_a[i]:
                vec_a = word_to_id(ans, vocab)
                feed_dict_1 = {model.input_x: input_q, model.input_y: vec_a,
                               model.keep_prob: config.DROPOUT_KEEP_PROB}
                feed_dict_2 = {model.input_x: vec_q, model.input_y: vec_a,
                               model.keep_prob: config.DROPOUT_KEEP_PROB}
                score1 = sess.run(model.result, feed_dict=feed_dict_1)
                # score2 = sess.run(config.model.result, feed_dict=feed_dict_2)
                score = score1

                temp.append((i, j, float(score)))
                j += 1
            temp.sort(key=lambda k: k[2], reverse=True)

            scores.append(temp[0])

        scores.sort(key=lambda k: k[2], reverse=True)
        row_id = scores[0][0]
        col_id = scores[0][1]

        answer = candidate_a[row_id][col_id][1]
        return answer
