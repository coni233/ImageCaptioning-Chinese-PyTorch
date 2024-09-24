import os

import torch
import torch.backends.cudnn as cudnn

image_h = image_w = image_size = 256 #图像的高度、宽度和大小
channel = 3
epochs = 10000 #训练的轮数？
patience = 10  #早停法的耐心值，如果验证集上的性能在 patience 个轮次内没有提升，则停止训练
num_train_samples = 1050000 #训练集的样本数量
num_valid_samples = 150000 #验证集的样本数量
max_len = 40 #最大句子长度
captions_per_image = 5 #每张图片的描述数量

# Model parameters
emb_dim = 512  # 词嵌入的维度
attention_dim = 512  # 注意力线性层的维度
decoder_dim = 512  # 解码器的 LSTM 层的维度
dropout = 0.5 #dropout概率，防止过拟合
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # 设置为 True 时，如果输入到模型的大小是固定的，可以加速训练；否则会有较大的计算开销。

# Training parameters
start_epoch = 0 #开始的轮次
epochs = 120  # 训练的轮数，设置为 120，如果没有触发早停法，则训练 120 个轮次。
epochs_since_improvement = 0  # 记录自上次验证集 BLEU 分数提升以来的轮次数
batch_size = 128
workers = 2  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # 编码器的学习率，如果进行微调，设置为 1e-4
decoder_lr = 4e-4  # 编码器的学习率
grad_clip = 5.  # 梯度裁剪的阈值，设置为 5，用于防止梯度爆炸
alpha_c = 1.  # 双重随机注意力机制的正则化参数，设置为 1
best_bleu4 = 0.  # 当前的 BLEU-4 分数，用于记录最佳模型的性能
print_freq = 100  # 每训练/验证 100 个批次打印一次统计信息
fine_tune_encoder = False  # 是否微调编码器，设置为 False 表示不进行微调
checkpoint = None  # 检查点的路径，用于加载和保存模型。如果没有检查点，则设置为 None
min_word_freq = 3 #词汇表中词的最小频率，低于该频率的词将被忽略

# Data parameters
data_folder = 'data'
train_folder = 'data/ai_challenger_caption_train_20170902'
valid_folder = 'data/ai_challenger_caption_validation_20170910'
test_a_folder = 'data/ai_challenger_caption_test_a_20180103'
test_b_folder = 'data/ai_challenger_caption_test_b_20180103'
train_image_folder = os.path.join(train_folder, 'caption_train_images_20170902')
valid_image_folder = os.path.join(valid_folder, 'caption_validation_images_20170910')
test_a_image_folder = os.path.join(test_a_folder, 'caption_test_a_images_20180103')
test_b_image_folder = os.path.join(test_b_folder, 'caption_test_b_images_20180103')
train_annotations_filename = os.path.join(train_folder, 'caption_train_annotations_20170902.json')
valid_annotations_filename = os.path.join(valid_folder, 'caption_validation_annotations_20170910.json')
test_a_annotations_filename = os.path.join(test_a_folder, 'caption_test_a_annotations_20180103.json')
test_b_annotations_filename = os.path.join(test_b_folder, 'caption_test_b_annotations_20180103.json')
