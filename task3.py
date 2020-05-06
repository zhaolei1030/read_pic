# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from numpy import array
import numpy as np
from os import listdir
from pickle import load
def load_doc(file_path):
    pic_name = []
    with open(file_path,'r') as doc:
        for line in doc:
            pic_name.append(line[:-5])
    return pic_name
def get_token(file_path):
    with open(file_path,'rb') as pkl:
        token = load(pkl)
    return token
def get_dir_list(dir_path):
    name = []
    total_name = listdir(dir_path)
    for i in total_name:
        name.append(i[:-4])
    return name

def create_input_data(token, max_length, descriptions, photos_features, vocab_size):
    """
    从输入的图片标题list和图片特征构造LSTM的一组输入

    Args:
    :param tokenizer: 英文单词和整数转换的工具keras.preprocessing.text.Tokenizer
    :param max_length: 训练数据集中最长的标题的长度
    :param descriptions: dict, key 为图像的名(不带.jpg后缀), value 为list, 包含一个图像的几个不同的描述
    :param photos_features:  dict, key 为图像的名(不带.jpg后缀), value 为numpy array 图像的特征
    :param vocab_size: 训练集中表的单词数量
    :return: tuple:
            第一个元素为 numpy array, 元素为图像的特征, 它本身也是 numpy.array
            第二个元素为 numpy array, 元素为图像标题的前缀, 它自身也是 numpy.array
            第三个元素为 numpy array, 元素为图像标题的下一个单词(根据图像特征和标题的前缀产生) 也为numpy.array
    """
    photo_seq = []
    des_sequence = []
    output_seq = []
    for key in descriptions:
        if descriptions[key] != []:
            seqs = token.texts_to_sequences(descriptions[key])
            for seq in seqs:
                for i in range(len(seq)-1):
                    des_sequence.append(np.pad(seq[:i+1],(max_length-i-1,0)))
                    photo_seq.append(photos_features[key])
                    zero = np.zeros(vocab_size)
                    zero[seq[i+1]] = 1
                    output_seq.append(zero)


    return np.array(photo_seq), np.array(des_sequence), np.array(output_seq)

if __name__ == "__main__":
    # name = load_doc('/Users/xiaopangzi/Desktop/dp code/homework2的副本2/task4/Flickr_8k.testImages.txt')
    # name = set(name)
    tokenizer = load(open('/Users/xiaopangzi/Desktop/dp code/homework2的副本2/task4/tokenizer.pkl', 'rb'))
    max_length = 6
    descriptions = {'1235345': ['startseq one bird on tree endseq', "startseq red bird on tree endseq"],
                    '1234546': ['startseq one boy play water endseq', "startseq one boy run across water endseq"]}
    photo_features = {'1235345': [0.434, 0.534, 0.212, 0.98],
                      '1234546': [0.534, 0.634, 0.712, 0.28]}
    vocab_size = 7378
    print(create_input_data(tokenizer, max_length, descriptions, photo_features, vocab_size))