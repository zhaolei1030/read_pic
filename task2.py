from keras.models import model_from_json
from PIL import Image as pil_image
from keras import backend as K
import numpy as np
from pickle import dump
from os import listdir
from keras.models import Model
import keras
import json as js


def load_vgg16_model():
    # with open('task5/vgg16_exported.json','r') as file:
    #     js_model = js.load(file)
    jsfile = open('task5/vgg16_exported.json','r')
    loaded_json = jsfile.read()
    jsfile.close()
    # js_model = str(js_model)
    model = model_from_json(loaded_json)

    model.load_weights('vgg16_exported.h5')
    """从当前目录下面的 vgg16_exported.json 和 vgg16_exported.h5 两个文件中导入 VGG16 网络并返回创建的网络模型
    # Returns
        创建的网络模型 model
    """
    return model


def preprocess_input(x):
    """预处理图像用于网络输入, 将图像由RGB格式转为BGR格式.
       将图像的每一个图像通道减去其均值

    # Arguments
        x: numpy 数组, 4维.
        data_format: Data format of the image array.

    # Returns
        Preprocessed Numpy array.
    """
    # for x0 in x[0]:
    #     for x1 in x0[0]:
    #         x1 -= np.array(x1)
    x = x[..., ::-1]
    print(x.shape)
    mean = [103,116,123]
    x[..., 0 ] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    return x


def load_img_as_np_array(path, target_size):
    """从给定文件加载图像,转换图像大小为给定target_size,返回32位浮点数numpy数组.

    # Arguments
        path: 图像文件路径
        target_size: 元组(图像高度, 图像宽度).

    # Returns
        A PIL Image instance.
    """
    image = pil_image.open(path)
    image = image.resize(target_size,pil_image.NEAREST)
    return np.asarray(image, dtype=K.floatx())


def extract_features(directory):
    """提取给定文件夹中所有图像的特征, 将提取的特征保存在文件features.pkl中,
       提取的特征保存在一个dict中, key为文件名(不带.jpg后缀), value为特征值[np.array]

    Args:
        directory: 包含jpg文件的文件夹

    Returns:
        None
    """
    model = load_vgg16_model()
    model.layers.pop()
    model = Model(input = model.inputs, output = model.layers[-1].output)
    features = {}
    i = 1
    for fn in listdir(directory):
        i += 1
        path = directory + '/' + fn
        pic = load_img_as_np_array(path, target_size=(224,224))
        # print(pic.shape)
        # print(type(pic))
        pic = pic.reshape((1, pic.shape[0], pic.shape[1], pic.shape[2]))
        arr = preprocess_input(pic)
        # print(type(arr))
        feature = model.predict(pic,verbose=0)
        id = fn[:-4]
        features[id] = feature
        if i >10:
            break
    return features

if __name__ == '__main__':
    # # 提取所有图像的特征，保存在一个文件中, 大约一小时的时间，最后的文件大小为127M
    # directory = '..\Flicker8k_Dataset'
    # features = extract_features(directory)
    # print('提取特征的文件个数：%d' % len(features))
    # print(keras.backend.image_data_format())
    # #保存特征到文件
    # dump(features, open('features.pkl', 'wb'))
    print(extract_features('Flicker8k_Dataset'))



