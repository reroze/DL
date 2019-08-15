#coding:utf-8
#全连接网络实例
import tensorflow as tf
import numpy as np
from PIL import Image
import backward
import forward

def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None,forward.INPUT_NODE])
        y = forward.forward(x,None)
        preValue = tf.argmax(y,1)#列表中概率最大的为输出结果

        variable_averages = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if(ckpt and ckpt.model_checkpoint_path):
                saver.restore(sess,ckpt.model_checkpoint_path)
                #print(sess.run(y), feed_dict={x:testPicArr})
                print(sess.run(y, feed_dict={x:testPicArr}))
                preValue = sess.run(preValue, feed_dict={x:testPicArr})
                #print(sess.run(y))

                return preValue
            else:
                print("No chekpoint file found")
                return -1


def pre_pic(picName):#图片预处理
    img = Image.open(picName)
    reIm = img.resize((28,28), Image.ANTIALIAS)#符合模型
    im_arr = np.array(reIm.convert('L'))#变成灰度图
    threshold = 50#灰度阈值
    for i in range(28):#反色
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if(im_arr[i][j] < threshold):#二值化处理
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)#变为浮点型
    img_ready = np.multiply(nm_arr, 1.0/255.0)

    return img_ready

def application():
    testNum = int(input("input the number of test pictures:"))
    for i in range(testNum):
        testPic = input("the path of test picture:")
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print("The prediction number is:", preValue)


def main():
    application()


if __name__ == '__main__':
    main()





