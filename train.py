# coding:utf-8

import os
import tensorflow as tf
from poems.poems import process_poems, generate_batch
from poems.model import rnn_model

tf.app.flags.DEFINE_integer('batch_size', 64, 'batch_size')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning_rate')
tf.app.flags.DEFINE_string('model_dir', os.path.abspath('./model'), 'model save path')
tf.app.flags.DEFINE_string('file_path', os.path.abspath('./data/poems.txt'), 'file name of poems')
tf.app.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix')
tf.app.flags.DEFINE_integer('epochs', 50, 'train how many epochs')

FLAGS = tf.app.flags.FLAGS


def run_training():
    # dir to save model

    if os.path.exists(FLAGS.model_dir):
        os.mkdir(FLAGS.model_dir)

    poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)
    batches_input, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)

    # print(word_to_int)
    # print(batches_input[0][0])
    # print(batches_outputs[0][1])
    # print(batches_outputs)
    # time.sleep(10000)

    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn_model(model="lstm", input_data=input_data, output_data=output_targets, vocab_size=len(vocabularies),
                           rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:

        # 初始化所有的变量
        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)

        if checkpoint:
            saver.restore(sess, checkpoint)
            print("### restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])

        print(' ## start training... ')

        try:

            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                n_chunk = len(poems_vector) // FLAGS.batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_input[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch, batch, loss))

                if epoch % 6:
                    saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            print('## Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
            print('## Last epoch were saved, next time will start from epoch {}.'.format(epoch))


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
