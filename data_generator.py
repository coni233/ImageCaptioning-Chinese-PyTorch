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

        # Total number of datapoints
        self.dataset_size = len(self.samples * captions_per_image)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform


    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        sample = self.samples[i // captions_per_image]
        path = os.path.join(self.image_folder, sample['image_id'])
        # Read images
        #img = imread(path) #弃用
        #img = Image.open(path).convert("RGB")
        # Read image
        img = imageio.imread(path)
        # Convert grayscale to RGB if necessary
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        # if len(img.shape) == 2:
        #     img = img[:, :, np.newaxis]
        #     img = np.concatenate([img, img, img], axis=2)
        #img = imresize(img, (256, 256))
        #img = img.resize((256, 256), Image.ANTIALIAS)
        # Convert to PIL Image to use PIL operations
        img = Image.fromarray(img)
        img = img.resize((256, 256), Image.LANCZOS)  # Use LANCZOS for resizing
        if self.transform:
            img = self.transform(img)
        #img = self.transform(img)
        # img = img.transpose(2, 0, 1)
        # assert img.shape == (3, 256, 256)
        # assert np.max(img) <= 255
        # img = torch.FloatTensor(img / 255.)
        # if self.transform is not None:
        #     img = self.transform(img)

        # Sample captions
        captions = sample['caption']
        # Sanity check
        assert len(captions) == captions_per_image
        c = captions[i % captions_per_image]
        c = list(jieba.cut(c))
        # Encode captions
        enc_c = encode_caption(self.word_map, c)

        caption = torch.LongTensor(enc_c)

        caplen = torch.LongTensor([len(c) + 2])

        if self.split == 'train':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor([encode_caption(self.word_map, list(jieba.cut(c))) for c in captions])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
