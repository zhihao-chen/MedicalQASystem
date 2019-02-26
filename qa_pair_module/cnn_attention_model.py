import tensorflow as tf


class CNNAttention(object):
    def __init__(self,
                 max_seg_length,
                 embeddings,
                 embedding_dim,
                 filter_size,
                 num_filter,
                 dropout_keep_prob,
                 margin_value,
                 learning_rate,
                 l2_reg):

        self.seg_length = max_seg_length
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.filter_size = filter_size
        self.num_filter = num_filter
        self.dropout_keep_prob = dropout_keep_prob
        self.margin_value = margin_value
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg

        # initializer placeholder
        self.input_question = tf.placeholder(tf.int32, [None, self.seg_length], name='input_q')
        self.input_true_answer = tf.placeholder(tf.int32, [None, self.seg_length], name='input_a_pos')
        self.input_false_answer = tf.placeholder(tf.int32, [None, self.seg_length], name='input_a_neg')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.input_x = tf.placeholder(tf.int32, [None, self.seg_length])
        self.input_y = tf.placeholder(tf.int32, [None, self.seg_length])

        # 设置embedding 层 [batch_size, seq_length, embedding_size]
        with tf.name_scope('embedding-layer'):
            W = tf.Variable(tf.to_float(self.embeddings), trainable=True, name='W')
            # W = tf.Variable(tf.random_uniform([10000, self.embedding_dim], -1.0, 1.0), name="W")
            q_embed = self.embedding_layer(W, self.input_question)
            answer_true_embed = self.embedding_layer(W, self.input_true_answer)
            answer_false_embed = self.embedding_layer(W, self.input_false_answer)

            test_q_embed = self.embedding_layer(W, self.input_x)
            test_answer_embed = self.embedding_layer(W, self.input_y)

        # attention 矩阵 [batch_size, embedding_size, embedding_size]
        with tf.name_scope('attention-layer'):
            att_mat_1 = self.attention_mat(q_embed, answer_true_embed)
            att_mat_2 = self.attention_mat(q_embed, answer_false_embed)
            #print(att_mat_2.shape)
            att_mat = self.attention_mat(test_q_embed, test_answer_embed)

        # [batch, d, d] * [d,s] => [batch, d, s]
        # matrix transpose => [batch, s, d]
        # expand dims => [batch, s, d, 1]
        aW = tf.get_variable(name="aW",
                             shape=(self.embedding_dim, self.seg_length),
                             initializer=tf.contrib.layers.xavier_initializer())
                             #regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg))
        f1_1, f1_2 = self.attention_feature_map(aW, att_mat_1)
        f2_1, f2_2 = self.attention_feature_map(aW, att_mat_2)

        f1, f2 = self.attention_feature_map(aW, att_mat)

        # [batch, s, d, 2]
        x1_q = tf.concat([q_embed, f1_1], axis=3)
        x1_a = tf.concat([answer_true_embed, f1_2], axis=3)
        x2_q = tf.concat([q_embed, f2_1], axis=3)
        x2_a = tf.concat([answer_false_embed, f2_2], axis=3)

        x_q = tf.concat([test_q_embed, f1], axis=3)
        x_a = tf.concat([test_answer_embed, f2], axis=3)

        # 卷积层
        with tf.name_scope('conv-layer'):
            q1_conved = self.cnn_layer(x1_q, reuse=False)
            true_answer = self.cnn_layer(x1_a, reuse=True)
            q2_conved = self.cnn_layer(x2_q, reuse=True)
            false_answer = self.cnn_layer(x2_a, reuse=True)

            test_q = self.cnn_layer(x_q, reuse=True)
            test_a = self.cnn_layer(x_a, reuse=True)

        # 输出层
        with tf.name_scope('output-layer'):
            pos_sim = self.getCosineSimilarity(q1_conved, true_answer)
            neg_sim = self.getCosineSimilarity(q2_conved, false_answer)

        # loss = max(0, m-sim1 + sim2)
        with tf.name_scope('loss'):
            shape = tf.shape(pos_sim)
            zero = tf.fill(shape, 0.0)
            margin = tf.fill(shape, self.margin_value)

            losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(pos_sim, neg_sim)))
            self.loss = tf.reduce_sum(losses)

        with tf.name_scope('accuracy'):
            correct = tf.equal(zero, losses)
            self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'), name='acc')

        self.result = self.getCosineSimilarity(test_q, test_a)

    def embedding_layer(self, w, x):
        embed = tf.nn.embedding_lookup(w, x)
        embedding = tf.expand_dims(embed, -1)
        return embedding

    def attention_mat(self, q, a):
        euclidean = tf.sqrt(tf.reduce_sum(tf.square(q - tf.matrix_transpose(a)), axis=1))
        return 1 / (1 + euclidean)

    def attention_feature_map(self, aW, att_mat):
        x1_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", att_mat, aW)), -1)
        x2_a = tf.expand_dims(tf.matrix_transpose(
            tf.einsum("ijk,kl->ijl", tf.matrix_transpose(att_mat), aW)), -1)
        return x1_a, x2_a

    def cnn_layer(self, x, reuse):
        pooled = []
        with tf.variable_scope('conv', reuse=tf.AUTO_REUSE):
            for k in self.filter_size:
                conv = tf.contrib.layers.conv2d(
                    inputs=x,
                    num_outputs=self.num_filter,
                    kernel_size=(k, self.embedding_dim),
                    stride=1,
                    padding="VALID",
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg),
                    biases_initializer=tf.constant_initializer(1e-04),
                    trainable=True
                )
            # output: [batch_size, s-k+1, 1, num_filter]

            # [batch, num_filter, s-k+1, 1]
                conv_trans = tf.transpose(conv, [0, 3, 1, 2], name="conv_trans-{}".format(k))
                m_ap = tf.contrib.layers.max_pool2d(conv_trans,
                                                    kernel_size=(1, self.seg_length-k+1),
                                                    stride=1,
                                                    padding='VALID')
            # m_ap:[batch, num_filter, 1, 1]

            # [batch, 1,num_filter, 1]
                map_trans = tf.transpose(m_ap, [0, 2, 1, 3])
                pooled.append(map_trans)
        # [batch, 1, 2*num_filter, 1]
        result = tf.concat(pooled, 2)
        total_filters = self.num_filter * len(self.filter_size)
        # [batch, 2*num_filter]
        pooled_flat = tf.reshape(result, [-1, total_filters])
        #pooled_flat_1 = tf.layers.batch_normalization(pooled_flat, axis=1)
        vec_drop = tf.nn.dropout(pooled_flat, self.keep_prob)
        return vec_drop

    def getCosineSimilarity(self, q, a):
        # 求模
        norm_q = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        norm_a = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))

        # 内积
        q_a = tf.reduce_sum(tf.multiply(q, a), 1)

        # cosine similarity
        cosSim = tf.div(q_a, tf.multiply(norm_q, norm_a))
        return cosSim
