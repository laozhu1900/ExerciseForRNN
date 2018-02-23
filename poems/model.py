# coding:utf-8

import tensorflow as tf


def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2,
              batch_size=64, learning_rate=0.01):
    """
    construct rnn seq2seq model
    :param model:
    :param input_data:
    :param output_data:
    :param vocab_size:
    :param rnn_size:
    :param num_layers:
    :param batch_size:
    :param learning_rate:
    :return:
    """

    end_points = {}

    if model == 'rnn':
        cell_fun = tf.nn.rnn_cell.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.nn.rnn_cell.GRUCell
    elif model == 'lstm':
        '''
        tf.nn.rnn_cell.BasicLSTMCell
        Args:

        num_units: int, The number of units in the LSTM cell.
        forget_bias: float, The bias added to forget gates (see above).
        state_is_tuple: If True, accepted and returned states are 2-tuples of the c_state and m_state. 
                        If False, they are concatenated along the column axis. 
                        The latter behavior will soon be deprecated.
        activation: Activation function of the inner states. Default: tanh.
        reuse: (optional) Python boolean describing whether to reuse variables in an existing scope. If not True,
                    and the existing scope already has the given variables, an error is raised.
            
        '''
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell
    else:
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell

    '''
       state_is_tuple:
       rnn_size=128
       num_layers=2
    '''
    cell = cell_fun(rnn_size, state_is_tuple=True)

    # state_is_tuple :参考图片中LSTM的解释,输出的是c和a

    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    if output_data is not None:
        initial_state = cell.zero_state(batch_size, tf.float32)

    else:
        initial_state = cell.zero_state(1, tf.float32)

    with tf.device("/cpu:0"):
        """
            tf.random_uniform　生成矩阵
            取值范围是：[-1.0, 1.0]
            行数：vocab_size + 1
            列数：rnn_size
        """
        embedding = tf.get_variable('embdding', initializer=tf.random_uniform(
            [vocab_size + 1, rnn_size], -1.0, 1.0))

        '''
            embedding_lookup :
                参考：http://blog.csdn.net/u013041398/article/details/60955847
        '''
        inputs = tf.nn.embedding_lookup(embedding, input_data)

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)

    # tf.reshape(元组，[行，列])
    output = tf.reshape(outputs, [-1, rnn_size])

    # truncated_normal 正太分布随机值
    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size + 1]))

    bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]))

    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)

    if output_data is not None:
        # one_hot:　某一行上某个索引为１，其他为０
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)

        # 　交叉熵验证
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        # 求平均值，　tf.reduce_mean(loss, 0) 0　每一列求平均值, 1每一行求平均值
        total_loss = tf.reduce_mean(loss)

        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else:
        prediction = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points
