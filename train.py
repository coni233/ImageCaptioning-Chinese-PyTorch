import json
import time

import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from config import *
from data_generator import CaptionDataset
from models import Encoder, DecoderWithAttention
from utils import *

import gc

def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    #如果没有提供检查点文件，则初始化新的解码器和编码器。
    if checkpoint is None:
        #初始化带有注意力机制的解码器 DecoderWithAttention，并传入注意力维度、嵌入维度、解码器维度、词汇表大小和 dropout 概率。
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        # decoder = nn.DataParallel(decoder) # 使用多 GPU 训练
        # decoder = torch.nn.DataParallel(decoder.cuda(), device_ids=[0, 1, 2, 3]) # 使用多 GPU 训练
        
        #使用 Adam 优化器优化解码器参数，学习率为 decoder_lr。
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        #设置是否需要微调编码器
        encoder.fine_tune(fine_tune_encoder)
        # encoder = nn.DataParallel(encoder)
        # encoder = torch.nn.DataParallel(encoder.cuda(), device_ids=[0, 1, 2, 3])
        #如果需要微调编码器，则使用 Adam 优化器优化编码器参数，学习率为 encoder_lr。
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        #从检查点文件中加载解码器、编码器、解码器优化器、编码器优化器、最佳 BLEU-4 分数和自上次提升以来的轮次数。
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    #交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
    transforms.ToTensor(),  # 首先将 PIL 图像转换为 PyTorch 张量，将像素值从 [0, 255] 缩放到 [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对张量进行标准化处理。mean 和 std 分别是每个通道的均值和标准差，用于将图像数据归一化到标准正态分布。
    ])

    #创建一个训练集数据加载器，用于批量加载数据。返回图像、字幕的序列，和字幕长度
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset('train', transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    #创建一个验证集数据加载器，用于批量加载数据。返回图像、字幕的序列、字幕长度和这幅图片所有（5个）字幕的序列
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset('valid', transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # 如果连续 20 个 epoch 没有改进，则终止训练
        if epochs_since_improvement == 20:
            break
        # 如果连续 8 个 epoch 没有改进，则衰减学习率
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader, #训练集数据加载器
              encoder=encoder, #编码器
              decoder=decoder, #解码器
              criterion=criterion, #损失函数
              encoder_optimizer=encoder_optimizer, #编码器优化器
              decoder_optimizer=decoder_optimizer, #解码器优化器
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # 检查是否有改进
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)
        
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清空 CUDA 缓存以释放内存


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: 训练数据的 DataLoader
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: 损失函数
    :param encoder_optimizer: 用于更新编码器权重的优化器（如果进行微调）
    :param decoder_optimizer: 用于更新解码器权重的优化器
    :param epoch: 当前的 epoch 编号
    """

    decoder.train()  # 训练模式（使用 dropout 和 batchnorm）
    encoder.train()

    batch_time = AverageMeter()  # 前向传播和反向传播时间
    data_time = AverageMeter() # 数据加载时间
    losses = AverageMeter()  # 每个单词解码的损失
    top5accs = AverageMeter()  # top5 准确率

    start = time.time()

    # 批次训练
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # 如果可用，移动到 GPU
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # 前向传播
        imgs = encoder(imgs)
        #解码器的输出是解码后的分数、解码后的字幕、解码后的长度、注意力权重和排序索引
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # 由于我们从 <start> 开始解码，目标是 <start> 之后的所有单词，直到 <end>
        targets = caps_sorted[:, 1:]

        # 移除我们没有解码的时间步或填充
        # pack_padded_sequence 是一个简单的技巧来做到这一点
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # 计算损失
        loss = criterion(scores.data, targets.data)

        # 添加双重随机注意力正则化，用于约束注意力权重的分布
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # 反向传播
        #清除优化器中的梯度，以确保每次反向传播时不会累积之前的梯度，计算当前损失相对于模型参数的梯度，为后续的参数更新做准备
        decoder_optimizer.zero_grad() #清除解码器优化器中存储的梯度
        if encoder_optimizer is not None: #检查编码器优化器是否存在
            encoder_optimizer.zero_grad() #如果存在，则清除编码器优化器中存储的梯度
        loss.backward() #计算损失相对于模型参数的梯度

        # 梯度裁剪
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # 更新权重
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # 记录指标

        top5 = accuracy(scores, targets, 5) #计算 top-5 准确率，表示预测的前 5 个结果中是否包含正确答案
        losses.update(loss.item(), sum(decode_lengths)) #更新损失值，loss.item() 返回当前损失的标量值，sum(decode_lengths) 表示解码的总长度
        top5accs.update(top5, sum(decode_lengths)) #更新 top-5 准确率
        batch_time.update(time.time() - start) #更新批次时间

        start = time.time()

        # 打印状态
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: 验证数据的 DataLoader
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: 损失函数
    :return: BLEU-4 分数
    """
    decoder.eval()  # 评估模式（无 dropout 或 batchnorm）
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # 用于计算 BLEU-4 分数的参考（真实字幕）
    hypotheses = list()  # 假设（预测）

    # 批次
    for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

        # 移动到设备（如果可用）
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        allcaps = allcaps.to(device)

        # 前向传播
        if encoder is not None:
            imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # 由于我们从 <start> 开始解码，目标是 <start> 之后的所有单词，直到 <end>
        targets = caps_sorted[:, 1:]

        # 移除我们没有解码的时间步或填充
        # pack_padded_sequence 是一个简单的技巧来做到这一点
        scores_copy = scores.clone()
        scores= pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets= pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # 计算损失
        loss = criterion(scores.data, targets.data)

        # 添加双重随机注意力正则化
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # 记录指标
        losses.update(loss.item(), sum(decode_lengths))
        top5 = accuracy(scores, targets, 5)
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                            loss=losses, top5=top5accs))

        # 存储每个图像的参考（真实字幕）和假设（预测）
        # 如果对于 n 个图像，我们有 n 个假设，并且每个图像有参考 a, b, c...，我们需要 -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

        # 参考 references
        allcaps = allcaps[sort_ind]  # 因为图像在解码器中被排序了
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist() #转换为 Python 列表
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                    img_caps))  # 移除 <start> 和 <pad> 标记
            references.append(img_captions)

        # 假设 hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # 移除填充
        preds = temp_preds
        hypotheses.extend(preds)

        assert len(references) == len(hypotheses)

    # 计算 BLEU-4 分数
    bleu4 = corpus_bleu(references, hypotheses)

    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
