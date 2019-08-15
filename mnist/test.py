#coding:utf-8
#用于测试mnist手写数据集的准确性
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward
import backward
import generateds

TEST_INTERVAL_SECS = 5
TEST_NUM = 10000#extre for : mnist.test.num_examples

#def test(mnist):#before
def test():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
        y = forward.forward(x, None)
        #喂入对应的x，y数据，使用forward文件中的forward函数搭建网络

        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        #滑动平均

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #正则化
        img_batch, label_batch = generateds.get_tfrecord(TEST_NUM, isTrain=False)#从测试集中选择数据，数据没有被训练过，应该填False

        while(True):
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
                print(ckpt.model_checkpoint_path)
                if (ckpt and ckpt.model_ckeckpoint_path):
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    #从ckpt.model_ckeckpoint_path中读取数据
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    coord = tf.train.Coordinator()#extre
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)#extre
                    xs, ys=sess.run([img_batch, label_batch])#extre


                    #before : accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels})
                    accuracy_score = sess.run(accuracy, feed_dict={x:xs, y:ys})
                    print("After %s training step(s), test accuracy = %g" %(global_step, accuracy_score))

                    coord.request_stop()#extre
                    coord.join(threads)#extre关闭多线程
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(TEST_INTERVAL_SECS)
            #延时函数


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    #定义mnist变量
    #test(mnist)#before
    test()#extre


if __name__ == '__main__':
    main()
