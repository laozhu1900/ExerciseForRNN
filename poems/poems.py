# coding:utf-8

import collections
import os
import sys
import numpy as np

start_token = 'B'
end_token = 'E'


def process_poems(file_name):
    # poems
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(":")
                content = content.replace(" ", "")
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue

                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                pass

    # 根据字符长度升序排列
    poems = sorted(poems, key=lambda l: len(line))

    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items())
    words, _ = zip(*count_pairs)

    words = words[:len(words)] + (' ',)

    '''
        some notice
        words = ['a','b','c','d'] 
        range(len(words)) = range(4) = [0,1,2,3]
        word_int_map = {"a":0, "b":1, "c":2, "d":3}
        poems_vector: 每个字符在map中的索引，[2,1,3,4]
    '''
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(lambda word: word_int_map.get(word, len(words)), poem)) for poem in poems]

    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size

    x_batches = []
    y_batches = []

    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = poems_vec[start_index:end_index]
        # print(batches)

        # 选择长度最大的诗词
        length = max(map(len, batches))

        '''
           np.full((row, columns), num) 用num填充一个row*column的矩阵
        '''
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        # print((batch_size, length), word_to_int[' '])
        # print(x_data)

        for row in range(batch_size):
            # 给矩阵赋值，将之前的训练集赋值给这个矩阵
            x_data[row, :len(batches[row])] = batches[row]

        y_data = np.copy(x_data)

        y_data[:, :-1] = x_data[:, 1:]

        """
            x中元素["B",'1','2','3'..."E"]
            y中元素['1','2','3'.....'E','E']
            目的是错开一个位置:y_data中是下一个值，x_data预测出来的中和ｙ进行计算差值
            x_data            y_data
            [6,2,4,6,9]       [2,4,6,9,9]
            [1,4,2,8,5]       [4,2,8,5,5]
        """

        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


if __name__ == '__main__':
    filename = "../data/test.dat"
    a, b, c = process_poems(filename)
    # print(a)
    # print(b)
    d, e = generate_batch(2, a, b)
    #
    print(d[0])
    print(d[0].shape)
    # print(len("寒随穷律变，春逐鸟声开。初风飘带柳，晚雪间花梅。碧林青旧竹，绿沼翠新苔。芝田初雁去，绮树巧莺来。"))
