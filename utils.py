import multiprocessing
import os

import cv2 as cv
import torch

from torch.nn.utils.rnn import pad_packed_sequence

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()


#在图像上绘制字符串
'''
dst：目标图像，即要在其上绘制文本的图像。
target：文本的位置，表示为 (x, y) 坐标。
s：要绘制的文本字符串。
'''
def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


#用于在反向传播过程中裁剪梯度，以避免梯度爆炸
'''
通过遍历优化器的参数组和参数，并检查每个参数的梯度是否存在，
函数将梯度限制在指定的范围内，从而确保训练过程的稳定性
'''
def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


#保存模型检查点
def save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: 当前的训练轮数。
    :param epochs_since_improvement: 自上次 BLEU-4 分数提高以来的轮数。
    :param encoder: 编码器模型。
    :param decoder: 解码器模型。
    :param encoder_optimizer: 用于更新编码器权重的优化器（如果在微调）。
    :param decoder_optimizer: 用于更新解码器权重的优化器。
    :param bleu4: 当前轮次的验证 BLEU-4 分数。
    :param is_best: 当前检查点是否是迄今为止最好的?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = f'models/checkpoint_epoch_{epoch}_bleu4_{bleu4:.2f}.pth.tar'
    #filename = 'checkpoint_' + '.pth.tar'
    torch.save(state, filename)
    # 如果当前检查点是最好的，则将其另存为一个特殊的文件，以免被后续的较差检查点覆盖。
    if is_best:
        filename_best = f'models/best_checkpoint.pth.tar'
        torch.save(state, filename_best)
        print(f"Saved new best checkpoint to '{filename_best}'")


#用于跟踪某个指标的最新值、平均值、总和和计数
class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    #重置所有跟踪的值。
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    #更新最新值、总和、计数和平均值。
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


#调整优化器的学习率,按指定的因子缩小学习率。
def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: 需要缩小学习率的优化器.
    :param shrink_factor: 用于乘以学习率的因子，该因子的取值范围在(0, 1)之间.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


#这个函数用于计算模型预测的 top-k 准确率
'''
Top-k 准确率是一种评估分类模型性能的指标，特别适用于多类分类问题。
它表示模型预测的前 k 个最可能的类别中是否包含了正确的类别。(光束搜索内味了)
具体来说，如果模型的预测结果中，正确的类别出现在前 k 个最高概率的类别中，则认为该预测是正确的。
'''
def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: 模型的预测分数
    :param targets: 真实标签
    :param k: top-k 准确率中的 k 值
    :return: top-k 准确率
    """
    if isinstance(scores, torch.nn.utils.rnn.PackedSequence):
        scores, _ = pad_packed_sequence(scores, batch_first=True)
    if isinstance(targets, torch.nn.utils.rnn.PackedSequence):
        targets, _ = pad_packed_sequence(targets, batch_first=True)

    # print("Batch size (scores):", scores.size(0))
    # print("Batch size (targets):", targets.size(0))
    # print("Scores shape:", scores.shape)
    # print("Targets shape:", targets.shape)

    #batch_size = targets.size(0)
    #_, ind = scores.topk(k, 1, True, True)
    batch_size, seq_len, num_classes = scores.size()
    _, topk_preds = scores.reshape(-1, num_classes).topk(k, dim=1, largest=True, sorted=True)  # 重塑并获取 top-k 预测

    # 重塑 targets 以匹配 topk_preds 的形状
    targets = targets.reshape(-1).unsqueeze(1).expand(-1, k)
    
    correct = topk_preds.eq(targets).sum()  # 计算正确预测的数量
    total = targets.size(0)  # 总预测数量

    return (correct.float().item() / total) * 100


    # correct = ind.eq(targets.reshape(-1, 1).expand_as(ind))
    # correct_total = correct.view(-1).float().sum()  # 0D tensor
    # return correct_total.item() * (100.0 / batch_size)
