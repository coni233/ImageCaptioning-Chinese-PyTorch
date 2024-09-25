# encoding=utf-8
import json

import jieba
import numpy as np
#from scipy.misc import imread, imresize
from PIL import Image
import imageio
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from config import *

#将句子编码为整数序列
#在字幕的开头添加<start>
#将字幕中的每个单词转换为对应的整数 ID。如果单词不在 word_map 中，则使用特殊的未知标记 <unk>
#在字幕的结尾添加一个特殊的结束标记 <end>
#字幕的末尾添加填充标记 <pad>，使得字幕的总长度达到 max_len
def encode_caption(word_map, c):
    return [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))


transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像从PIL格式转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化张量图像
])

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, split, transform=None):
        """
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'train', 'valid'}

        if split == 'train':
            json_path = train_annotations_filename
            self.image_folder = train_image_folder
        else:
            json_path = valid_annotations_filename
            self.image_folder = valid_image_folder

        # Read JSON
        with open(json_path, 'r') as j:
            self.samples = json.load(j)

        # Read wordmap
        with open(os.path.join(data_folder, 'WORDMAP.json'), 'r') as j:
            self.word_map = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        #self.transform = transform

        # 计算数据集的大小
        self.dataset_size = len(self.samples * captions_per_image)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    #用于根据索引 i 获取数据集中的一个样本（图像及其对应的字幕）
    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        #计算第 i 个字幕对应的图像索引
        #sample是一个dict，里面存放了图片的id和5个句子
        sample = self.samples[i // captions_per_image]
        #构建图像文件的完整路径
        path = os.path.join(self.image_folder, sample['image_id'])
        # Read images
        #img = imread(path) #弃用
        #img = Image.open(path).convert("RGB")
        # Read image
        img = imageio.imread(path)
        # 将灰度图像转换为 RGB
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        # numpy转换为PIL image
        img = Image.fromarray(img)
        # 调整图像大小
        img = img.resize((256, 256), Image.LANCZOS)  # 使用 LANCZOS 算法调整图像大小
        #应用图像变换
        if self.transform:
            img = self.transform(img)

        # 获取图像对应的所有字幕
        captions = sample['caption']
        # Sanity check
        assert len(captions) == captions_per_image
        c = captions[i % captions_per_image]
        c = list(jieba.cut(c))
        # Encode captions
        #将字幕编码为整数序列，返回一个列表，列表中的每个元素是一个整数，代表字幕中的一个单词
        enc_c = encode_caption(self.word_map, c)
        #将 enc_c 转换为 PyTorch 的 LongTensor 类型
        caption = torch.LongTensor(enc_c)
        #将字幕的长度（加上起始和结束标记的长度）转换为 PyTorch 的 LongTensor 类型
        caplen = torch.LongTensor([len(c) + 2])

        if self.split == 'train':
            #训练时，返回图像、字幕的序列，和字幕长度
            return img, caption, caplen
        else:
            # 为了验证测试，还返回所有“captions_per_image”字幕以查找BLEU-4分数
            #验证时，返回图像、字幕的序列、字幕长度和这幅图片所有（5个）字幕的序列
            all_captions = torch.LongTensor([encode_caption(self.word_map, list(jieba.cut(c))) for c in captions])
            return img, caption, caplen, all_captions

    #返回对象的长度，用于返回数据集的大小
    def __len__(self):
        return self.dataset_size
