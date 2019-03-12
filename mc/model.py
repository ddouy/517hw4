import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import GRUCell


def cbow_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        print("x len" + str(x_len))
        print("q len" + str(q_len))
        print("x max len" + str(JX))
        print("q max len" + str(JQ))

        # emb_mat = tf.get_variable('emb_mat', shape=[V, d])
        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        #(?, 15, 50)
        print(qq.shape)
        #(?, 15)
        print(q_mask.shape)

        qq_avg = tf.reduce_mean(bool_mask(qq, q_mask, expand=True), axis=1)  # [N, d]
        qq_avg_exp = tf.expand_dims(qq_avg, axis=1)  # [N, 1, d]
        qq_avg_tiled = tf.tile(qq_avg_exp, [1, JX, 1])  # [N, JX, d]

        print("start")
        print(xx.shape)
        print(qq_avg_tiled)

        xq = tf.concat([xx, qq_avg_tiled, xx * qq_avg_tiled], axis=2)  # [N, JX, 3d]
        xq_flat = tf.reshape(xq, [-1, 3*d])  # [N * JX, 3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs


def rnn_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        #first x_len is true, maxlen is JX
        x_mask = tf.sequence_mask(x_len, JX)
        #first q_len is true, maxlen is JQ
        q_mask = tf.sequence_mask(q_len, JQ)

        # emb_mat = tf.get_variable('emb_mat', shape=[V, d])
        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        p = config.keep_prob

        xx_fw_rnn_cell = GRUCell(num_units=d)
        xx_fw_rnn_cell = DropoutWrapper(xx_fw_rnn_cell, input_keep_prob=p)
        xx_bw_rnn_cell = GRUCell(num_units=d)
        xx_bw_rnn_cell = DropoutWrapper(xx_bw_rnn_cell, input_keep_prob=p)

        qq_fw_rnn_cell = GRUCell(num_units=d)
        qq_fw_rnn_cell = DropoutWrapper(qq_fw_rnn_cell, input_keep_prob=p)
        qq_bw_rnn_cell = GRUCell(num_units=d)
        qq_bw_rnn_cell = DropoutWrapper(qq_bw_rnn_cell, input_keep_prob=p)
        with tf.variable_scope("qqq"):
        	(qq_fw_out, qq_bw_out), _ = tf.nn.bidirectional_dynamic_rnn(qq_fw_rnn_cell, qq_bw_rnn_cell, qq, dtype=tf.float32)
        with tf.variable_scope("xxx"):
        	(xx_fw_out, xx_bw_out), _ = tf.nn.bidirectional_dynamic_rnn(xx_fw_rnn_cell, xx_bw_rnn_cell, xx, dtype=tf.float32)
        # print(inputs)
        # print("JX" + str(JX))
        # print("JQ" + str(JQ))
        # print(q.shape)
        # #(?, 17, 50)
        # print(qq_out)
        # #(?, 17)
        # print(q_mask.shape)
        qq_out = tf.concat([qq_fw_out, qq_bw_out], 2)
        # qq_out = tf.nn.dropout(qq_out, p)
        xx_out = tf.concat([xx_fw_out, xx_bw_out], 2)
        # xx_out = tf.nn.dropout(xx_out, p)

        qq_avg = tf.reduce_mean(bool_mask(qq_out, q_mask, expand=True), axis=1)  # [N, d]
        qq_avg_exp = tf.expand_dims(qq_avg, axis=1)  # [N, 1, d]
        qq_avg_tiled = tf.tile(qq_avg_exp, [1, JX, 1])  # [N, JX, d]

        # xx_rnn = tf.nn.bidirectional_dynamic_rnn(xxGRU, qqGRU, xx, dtype=tf.float32)

        # print("xx_out shape")
        # print(xx_out)

        # print(qq_avg_tiled)

        # print(xx_out * qq_avg_tiled)
        # print("lala")
        xq = tf.concat([xx_out, qq_avg_tiled, xx_out * qq_avg_tiled], axis=2)  # [N, JX, 3d]
        xq_flat = tf.reshape(xq, [-1, 6*d])  # [N * JX, 3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs

def attention_forward(config, inputs, scope=None):
    raise NotImplementedError()


def get_loss(config, inputs, outputs, scope=None):
    with tf.name_scope(scope or "loss"):
        y1, y2 = inputs['y1'], inputs['y2']
        logits1, logits2 = outputs['logits1'], outputs['logits2']
        loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y1, logits=logits1))
        loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y2, logits=logits2))
        loss = loss1 + loss2
        acc1 = tf.reduce_mean(tf.cast(tf.equal(y1, tf.cast(tf.argmax(logits1, 1), 'int32')), 'float'))
        acc2 = tf.reduce_mean(tf.cast(tf.equal(y2, tf.cast(tf.argmax(logits2, 1), 'int32')), 'float'))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('acc1', acc1)
        tf.summary.scalar('acc2', acc2)
        return loss


def exp_mask(val, mask, expand=False):
    if expand:
        mask = tf.expand_dims(mask, -1)
    return val - (1.0 - tf.cast(mask, 'float')) * 10.0e10


def bool_mask(val, mask, expand=False):
    if expand:
        mask = tf.expand_dims(mask, -1)
    return val * tf.cast(mask, 'float')
