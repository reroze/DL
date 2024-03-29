#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward
import os
import generateds

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY=0.99
MODEL_SAVE_PATH="./model/"
MODEL_NAME="mnist_model"
train_num_examples = 60000#手动给出训练的总样本数 原来的是由mnist.train.num.examples给出


#def backward(mnist):
def backward():

    x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
    y = forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        #before:mnist.train.num_examples / BATCH_SIZE,
        #extre
        train_num_examples / BATCH_SIZE,
        #extre end
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    #extre:
    img_batch, label_batch = generateds.get_tfrecord(BATCH_SIZE, isTrain=True)#新的喂食函数
    #extre end
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if(ckpt and ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)

        #extre:
        coord = tf.train.Coordinator()#多线程协调器
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #extre end

        for i in range(STEPS):
            #before:xs, ys = mnist.train.next_batch(BATCH_SIZE)
            #extre:
            xs, ys = sess.run([img_batch, label_batch])#使用计算来获取batch
            #extre end
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000==0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        #extre:
        coord.request_stop()
        coord.join(threads)#关闭线程协调器
        #extre end

def main():
    #before:
    #mnist = input_data.read_data_sets("./data/", one_hot=True)
    #backward(mnist)
    #extre
    backward()
    #extre end


if __name__ == '__main__':
    main()