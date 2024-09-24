#创建WORDMAP.json
import json
import zipfile
from collections import Counter, OrderedDict

import jieba
from tqdm import tqdm

from config import *
from utils import ensure_folder


def extract(folder):
    filename = '{}.zip'.format(folder)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')

#创建WORDMAP.json
def create_input_files():
    json_path = train_annotations_filename

    # Read JSON
    with open(json_path, 'r') as j:
        samples = json.load(j) #samples是一个list，里面存放了很多dict，每个dict里面存放了一个图片的信息，包括图片的id和5个句子

    # Read image paths and captions for each image
    word_freq = Counter() #统计词频，Counter是一个简单的计数器，用于统计字符出现的个数，返回一个字典，key是字符，value是字符出现的次数，

    for sample in tqdm(samples):
        caption = sample['caption'] #caption是一个list，里面存放了5个句子，
        #遍历每个句子
        for c in caption: 
            seg_list = jieba.cut(c, cut_all=True) #jieba.cut是一个生成器，返回的是一个分词后的list
            # Update word frequency
            word_freq.update(seg_list) #更新词频，word_freq里的字典中的key是词，value是词出现的次数

    # Create word map
    #将词频大于等于min_word_freq=3的词加入到words中
    words = [w for w in word_freq.keys() if word_freq[w] >= min_word_freq] #word_freq.keys()返回的是一个迭代器，里面存放了所有的词
    #将这行代码的作用是创建一个词汇表（word_map），将每个词映射到一个唯一的整数 ID。具体来说，它使用 enumerate 函数为每个词分配一个从 1 开始的 ID。
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1 #unk是未知词
    word_map['<start>'] = len(word_map) + 1 #start是句子的开始
    word_map['<end>'] = len(word_map) + 1 #end是句子的结束
    word_map['<pad>'] = 0 #pad是填充，这里的0是填充的ID

    print(len(word_map))
    print(words[:10])

    sorted_word_map = OrderedDict(sorted(word_map.items(), key=lambda item: item[1]))
    # Save word map to a JSON
    with open(os.path.join(data_folder, 'WORDMAP.json'), 'w') as j:
        json.dump(sorted_word_map, j, sort_keys=False) 


if __name__ == '__main__':
    # parameters
    ensure_folder('data')

    if not os.path.isdir(train_image_folder):
        extract(train_folder)

    if not os.path.isdir(valid_image_folder):
        extract(valid_folder)

    if not os.path.isdir(test_a_image_folder):
        extract(test_a_folder)

    if not os.path.isdir(test_b_image_folder):
        extract(test_b_folder)

    create_input_files()
