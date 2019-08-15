#coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os

image_train_path = "./mnist_data_jpg/mnist_train_jpg_60000/"#训练用数据集的测试路径
label_train_path = "./mnist_data_jpg/mnist_train_jpg_60000.txt"#文字形式的路径
tfRecord_train = "./data/mnist_train.tfrecords"#训练时的取样
image_test_path = "./mnist_data_jpg/mnist_test_jpg_10000/"#测试集图片路径
label_test_path = "./mnist_data_jpg/mnist_test_jpg_10000.txt"#标签字典
tfRecord_test="./data/mnist_test.tfrecords"#测试时的取样
data_path = "./data"
resize_height = 28
resize_width = 28

def write_tfRecord(tfRecordName, image_path, label_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0#计数器
    f = open(label_path, "r")#打开txt标签文件（包含图片路径和标签）
    contents = f.readlines()#按行读取文件中的内容
    f.close()
    for content in contents:
        value = content.split()#使用空格隔开每一行的内容
        img_path = image_path + value[0]#每一行前面的部分为图面的文件名称
        print(img_path)
        img = Image.open(img_path)
        img_raw = img.tobytes()#变成二进制文件
        labels = [0]*10#初始化标签数组
        labels[int(value[1])] = 1#每一行后面一位表示的是该照片对应的图片标签

        example = tf.train.Example(feature=tf.train.Features(feature={
            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        }))#制作字典
        writer.write(example.SeriallizerToString())
        num_pic += 1
        print("the number of picture:", num_pic)
    writer.close()
    print("write tfrecord successful")


def generate_tfRecord():
    isExists = os.path.exists(data_path)
    print(data_path)
    if not isExists:
        os.makedirs(data_path)
        print("The directory was created successfully")
    else:
        print("directory already exists")
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)

def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#解序列化
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           "label": tf.FixedLenFeature([10], tf.int64),#分10类
                                           "img_raw": tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(features["img_raw"], tf.uint8)#8位无符号整形
    img.set_shape([784])#1行7884列
    img = tf.cast(img, tf.float32) * (1./255)#变位浮点数形式
    label = tf.cast(features["label"], tf.float32)
    return img, label

def get_tfrecord(num, isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    img, label = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=num,
                                                    num_threads=2,#两个线程
                                                    capacity=1000,
                                                    min_after_dequeue=700)
    return img_batch, label_batch




def main():
    generate_tfRecord()


if __name__ == '__main__':
    main();